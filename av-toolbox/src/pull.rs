use std::fs::File;
use std::io::{self, BufRead, IsTerminal, Seek, SeekFrom};
use std::path::PathBuf;
use std::process::ExitCode;

use av_toolbox_models::{InstallError, Registry, Store, registry};

use crate::cli::{Args, PullArgs};
use crate::report;

const REPOSITORY: &str = "ChillFish8/av-toolbox";
const USER_AGENT: &str = concat!("av-toolbox/", env!("CARGO_PKG_VERSION"));

#[derive(Debug, thiserror::Error)]
pub(crate) enum PullError {
    #[error("there is no pack named `{identifier}`. The packs are: {known}")]
    UnknownPack { identifier: String, known: String },

    #[error("neither this build's release nor the latest release, {tag}, has {asset}")]
    NotReleased { tag: String, asset: String },

    #[error("the {tag} release has no {identifier} pack, and {REPOSITORY} has no published releases yet")]
    NoReleases { tag: String, identifier: String },

    #[error("downloading {url} failed: {reason}")]
    Download { url: String, reason: String },

    #[error("the pack could not be installed: {0}")]
    Install(#[from] InstallError),

    #[error(transparent)]
    Io(#[from] io::Error),
}

/// Whether a download found anything.
pub(crate) enum Fetched {
    Written,
    NotFound,
}

pub(crate) trait Fetch {
    /// Streams `url` into `into`, or reports that nothing is there.
    fn download(&self, url: &str, into: &mut File) -> Result<Fetched, PullError>;

    /// The tag of the repository's latest release, or nothing when it has no releases.
    fn latest_tag(&self) -> Result<Option<String>, PullError>;
}

pub(crate) trait Confirm {
    /// Whether to delete the `count` other installed versions of `identifier`.
    fn delete_old(&self, identifier: &str, count: usize) -> bool;
}

#[derive(Debug)]
pub(crate) enum Outcome {
    AlreadyInstalled(PathBuf),
    Installed {
        path: PathBuf,
        tag: String,
        pruned: Vec<String>,
    },
}

pub(crate) fn run(args: &Args, pull_args: &PullArgs) -> ExitCode {
    let mut stderr = report::stderr(args.color);
    let store = Store::from_environment();
    let registry = registry();
    let outcome = pull(&store, registry, &UreqFetch, &TerminalConfirm, pull_args);

    match outcome {
        Ok(Outcome::AlreadyInstalled(path)) => {
            let shown = path.display();
            let identifier = &pull_args.identifier;
            eprintln!("{identifier} is already installed at {shown}. Pass --force to download it again");
            ExitCode::SUCCESS
        },
        Ok(Outcome::Installed { path, tag, pruned }) => {
            let shown = path.display();
            eprintln!("installed {} {tag} to {shown}", pull_args.identifier);

            for old in &pruned {
                eprintln!("removed {} {old}", pull_args.identifier);
            }

            ExitCode::SUCCESS
        },
        Err(error) => {
            let message = error.to_string();
            report::error_line(&mut stderr, &message);
            ExitCode::FAILURE
        },
    }
}

/// Downloads and installs a pack, then prunes its other versions if told to or allowed to.
///
/// The pack comes from the store's own release, or from the latest release when that one has none.
pub(crate) fn pull(
    store: &Store,
    registry: &Registry,
    fetch: &dyn Fetch,
    confirm: &dyn Confirm,
    args: &PullArgs,
) -> Result<Outcome, PullError> {
    let identifier = args.identifier.as_str();

    if registry.pack(identifier).is_none() {
        let names: Vec<&str> = registry.identifiers().collect();
        let known = names.join(", ");

        return Err(PullError::UnknownPack {
            identifier: identifier.to_string(),
            known,
        });
    }

    let own_tag = store.tag().to_string();
    let own_dir = store.pack_dir(&own_tag, identifier);
    if own_dir.is_dir() && !args.force {
        return Ok(Outcome::AlreadyInstalled(own_dir));
    }

    let root = store.root();
    std::fs::create_dir_all(root)?;
    let mut download = tempfile::tempfile_in(root)?;
    let own_url = release_url(&own_tag, identifier);
    let fetched = fetch.download(&own_url, &mut download)?;

    let tag = match fetched {
        Fetched::Written => own_tag,
        Fetched::NotFound => {
            tracing::warn!("the {own_tag} release has no {identifier} pack, looking for the latest release");

            let latest_found = fetch.latest_tag()?;
            let Some(latest) = latest_found else {
                return Err(PullError::NoReleases {
                    tag: own_tag,
                    identifier: identifier.to_string(),
                });
            };
            tracing::warn!("using the latest release, {latest}");

            let latest_dir = store.pack_dir(&latest, identifier);
            if latest_dir.is_dir() && !args.force {
                return Ok(Outcome::AlreadyInstalled(latest_dir));
            }

            download.set_len(0)?;
            download.seek(SeekFrom::Start(0))?;

            let latest_url = release_url(&latest, identifier);
            let retried = fetch.download(&latest_url, &mut download)?;

            if let Fetched::NotFound = retried {
                return Err(PullError::NotReleased {
                    tag: latest,
                    asset: asset_name(identifier),
                });
            }

            latest
        },
    };

    download.seek(SeekFrom::Start(0))?;
    let path = store.install(registry, identifier, &tag, &mut download)?;

    let others = store.other_tags(identifier, &tag)?;
    let count = others.len();
    let should_prune = count > 0 && (args.prune || confirm.delete_old(identifier, count));
    let pruned = if should_prune {
        store.prune(identifier, &tag)?
    } else {
        Vec::new()
    };

    Ok(Outcome::Installed { path, tag, pruned })
}

fn asset_name(identifier: &str) -> String {
    format!("optional-dependency-{identifier}.zip")
}

fn release_url(tag: &str, identifier: &str) -> String {
    let asset = asset_name(identifier);
    format!("https://github.com/{REPOSITORY}/releases/download/{tag}/{asset}")
}

/// Whether an answer to a yes or no question means yes.
fn accepts(answer: &str) -> bool {
    let trimmed = answer.trim();
    trimmed.eq_ignore_ascii_case("y") || trimmed.eq_ignore_ascii_case("yes")
}

#[derive(serde::Deserialize)]
struct Release {
    tag_name: String,
}

/// Reads the tag out of a GitHub release fetched from `url`.
fn latest_tag_from(url: &str, text: &str) -> Result<String, PullError> {
    let parsed = serde_json::from_str::<Release>(text);

    match parsed {
        Ok(release) => Ok(release.tag_name),
        Err(error) => Err(PullError::Download {
            url: url.to_string(),
            reason: format!("the release could not be read: {error}"),
        }),
    }
}

fn download_failed(url: &str, error: &ureq::Error) -> PullError {
    PullError::Download {
        url: url.to_string(),
        reason: error.to_string(),
    }
}

/// Fetches from GitHub over HTTPS.
struct UreqFetch;

impl Fetch for UreqFetch {
    fn download(&self, url: &str, into: &mut File) -> Result<Fetched, PullError> {
        let request = ureq::get(url).header("User-Agent", USER_AGENT);
        let called = request.call();

        let mut response = match called {
            Ok(response) => response,
            Err(ureq::Error::StatusCode(404)) => return Ok(Fetched::NotFound),
            Err(error) => return Err(download_failed(url, &error)),
        };

        let mut body = response.body_mut().as_reader();
        io::copy(&mut body, into)?;

        Ok(Fetched::Written)
    }

    fn latest_tag(&self) -> Result<Option<String>, PullError> {
        let url = format!("https://api.github.com/repos/{REPOSITORY}/releases/latest");
        let request = ureq::get(&url)
            .header("User-Agent", USER_AGENT)
            .header("Accept", "application/vnd.github+json");
        let called = request.call();

        let mut response = match called {
            Ok(response) => response,
            Err(ureq::Error::StatusCode(404)) => return Ok(None),
            Err(error) => return Err(download_failed(&url, &error)),
        };

        let read = response.body_mut().read_to_string();
        let text = match read {
            Ok(text) => text,
            Err(error) => return Err(download_failed(&url, &error)),
        };

        let tag = latest_tag_from(&url, &text)?;
        Ok(Some(tag))
    }
}

/// Asks on the terminal, and never asks without one.
struct TerminalConfirm;

impl Confirm for TerminalConfirm {
    fn delete_old(&self, identifier: &str, count: usize) -> bool {
        let stdin = io::stdin();
        if !stdin.is_terminal() {
            return false;
        }

        eprint!("Delete {count} old versions of {identifier}? [y/N] ");

        let mut answer = String::new();
        let read = stdin.lock().read_line(&mut answer);
        if read.is_err() {
            return false;
        }

        accepts(&answer)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::collections::HashMap;
    use std::io::{Cursor, Write};

    use av_toolbox_models::sha256_hex;
    use zip::write::SimpleFileOptions;

    use super::*;

    const WEIGHTS: &[u8] = b"weights";

    struct FakeReleases {
        assets: HashMap<String, Vec<u8>>,
        latest: Option<String>,
        requested: RefCell<Vec<String>>,
    }

    impl FakeReleases {
        fn with(assets: &[(String, Vec<u8>)], latest: &str) -> Self {
            let assets = assets.iter().cloned().collect();

            Self {
                assets,
                latest: Some(latest.to_string()),
                requested: RefCell::new(Vec::new()),
            }
        }

        fn without_releases() -> Self {
            Self {
                assets: HashMap::new(),
                latest: None,
                requested: RefCell::new(Vec::new()),
            }
        }
    }

    impl Fetch for FakeReleases {
        fn download(&self, url: &str, into: &mut File) -> Result<Fetched, PullError> {
            self.requested.borrow_mut().push(url.to_string());

            let Some(bytes) = self.assets.get(url) else {
                return Ok(Fetched::NotFound);
            };

            into.write_all(bytes)?;
            Ok(Fetched::Written)
        }

        fn latest_tag(&self) -> Result<Option<String>, PullError> {
            Ok(self.latest.clone())
        }
    }

    struct Answer {
        delete: bool,
        asked: Cell<bool>,
    }

    impl Answer {
        fn of(delete: bool) -> Self {
            Self {
                delete,
                asked: Cell::new(false),
            }
        }
    }

    impl Confirm for Answer {
        fn delete_old(&self, _identifier: &str, _count: usize) -> bool {
            self.asked.set(true);
            self.delete
        }
    }

    fn store_in(temp: &tempfile::TempDir, tag: &str) -> Store {
        let root = temp.path().to_path_buf();
        Store::new(root, tag)
    }

    fn fake_registry() -> Registry {
        let hash = sha256_hex(WEIGHTS);
        let lock = format!("[fake]\n\"w.bin\" = \"{hash}\"\n");

        Registry::parse(&lock).expect("valid")
    }

    fn pack_zip(content: &[u8]) -> Vec<u8> {
        let buffer = Cursor::new(Vec::new());
        let mut writer = zip::ZipWriter::new(buffer);
        writer
            .start_file("w.bin", SimpleFileOptions::default())
            .expect("started");
        writer.write_all(content).expect("written");

        let finished = writer.finish().expect("finished");
        finished.into_inner()
    }

    fn args(prune: bool, force: bool) -> PullArgs {
        PullArgs {
            identifier: "fake".to_string(),
            prune,
            force,
        }
    }

    fn url(tag: &str) -> String {
        release_url(tag, "fake")
    }

    #[test]
    fn the_release_url_names_the_tag_and_the_asset() {
        let built = release_url("v0.1.0", "artcnn");

        assert_eq!(
            built,
            "https://github.com/ChillFish8/av-toolbox/releases/download/v0.1.0/optional-dependency-artcnn.zip"
        );
    }

    #[test]
    fn the_models_tag_matches_this_binary() {
        let expected = format!("v{}", env!("CARGO_PKG_VERSION"));

        assert_eq!(av_toolbox_models::TAG, expected);
    }

    #[test]
    fn an_unknown_pack_is_refused_naming_the_known_ones() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let store = store_in(&temp, "v0.1.0");
        let releases = FakeReleases::with(&[], "v0.1.0");
        let answer = Answer::of(false);
        let registry = fake_registry();
        let unknown = PullArgs {
            identifier: "nothing".to_string(),
            prune: false,
            force: false,
        };

        let error = pull(&store, &registry, &releases, &answer, &unknown).expect_err("refused");

        let message = error.to_string();
        assert!(
            message.contains("`nothing`") && message.contains("fake"),
            "{message}"
        );
        assert!(releases.requested.borrow().is_empty());
    }

    #[test]
    fn a_pack_installs_from_the_builds_own_release_creating_the_root() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let root = temp.path().join("not-yet");
        let store = Store::new(root.clone(), "v9.9.9");
        let assets = [(url("v9.9.9"), pack_zip(WEIGHTS))];
        let releases = FakeReleases::with(&assets, "v1.0.0");
        let answer = Answer::of(false);
        let registry = fake_registry();

        let request = args(false, false);
        let outcome = pull(&store, &registry, &releases, &answer, &request).expect("pulled");

        let expected_dir = root.join("v9.9.9").join("fake");
        assert!(
            matches!(&outcome, Outcome::Installed { path, tag, .. } if *path == expected_dir && tag == "v9.9.9")
        );
        assert_eq!(*releases.requested.borrow(), [url("v9.9.9")]);
    }

    #[test]
    fn a_missing_release_falls_back_to_the_latest() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let store = store_in(&temp, "v9.9.9");
        let assets = [(url("v1.0.0"), pack_zip(WEIGHTS))];
        let releases = FakeReleases::with(&assets, "v1.0.0");
        let answer = Answer::of(false);
        let registry = fake_registry();

        let request = args(false, false);
        let outcome = pull(&store, &registry, &releases, &answer, &request).expect("pulled");

        assert!(matches!(&outcome, Outcome::Installed { tag, .. } if tag == "v1.0.0"));
        assert_eq!(*releases.requested.borrow(), [url("v9.9.9"), url("v1.0.0")]);
        assert!(store.pack_dir("v1.0.0", "fake").join("w.bin").is_file());
    }

    #[test]
    fn a_missing_release_with_no_releases_at_all_is_refused_and_leaves_nothing() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let store = store_in(&temp, "v9.9.9");
        let releases = FakeReleases::without_releases();
        let answer = Answer::of(false);
        let registry = fake_registry();

        let request = args(false, false);
        let error = pull(&store, &registry, &releases, &answer, &request).expect_err("refused");

        let message = error.to_string();
        let temp_path = temp.path();
        let listing = std::fs::read_dir(temp_path).expect("listed");
        assert!(matches!(&error, PullError::NoReleases { .. }), "{error}");
        assert!(
            message.contains("v9.9.9") && message.contains("fake"),
            "{message}"
        );
        assert_eq!(listing.count(), 0);
    }

    #[test]
    fn neither_release_holding_the_pack_is_refused() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let store = store_in(&temp, "v9.9.9");
        let releases = FakeReleases::with(&[], "v1.0.0");
        let answer = Answer::of(false);
        let registry = fake_registry();

        let request = args(false, false);
        let error = pull(&store, &registry, &releases, &answer, &request).expect_err("refused");

        assert!(
            matches!(&error, PullError::NotReleased { tag, .. } if tag == "v1.0.0"),
            "{error}"
        );
    }

    #[test]
    fn an_installed_pack_is_not_downloaded_again_unless_forced() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let store = store_in(&temp, "v9.9.9");
        let assets = [(url("v9.9.9"), pack_zip(WEIGHTS))];
        let releases = FakeReleases::with(&assets, "v9.9.9");
        let answer = Answer::of(false);
        let registry = fake_registry();
        let plain = args(false, false);
        let forcing = args(false, true);
        pull(&store, &registry, &releases, &answer, &plain).expect("pulled");

        let again = pull(&store, &registry, &releases, &answer, &plain).expect("no-op");
        let forced = pull(&store, &registry, &releases, &answer, &forcing).expect("pulled");

        assert!(matches!(again, Outcome::AlreadyInstalled(_)));
        assert!(matches!(forced, Outcome::Installed { .. }));
        assert_eq!(releases.requested.borrow().len(), 2);
    }

    #[test]
    fn a_fallback_to_an_installed_latest_is_not_downloaded_again() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let store = store_in(&temp, "v9.9.9");
        let assets = [(url("v1.0.0"), pack_zip(WEIGHTS))];
        let releases = FakeReleases::with(&assets, "v1.0.0");
        let answer = Answer::of(false);
        let registry = fake_registry();
        let request = args(false, false);
        pull(&store, &registry, &releases, &answer, &request).expect("pulled");

        let again = pull(&store, &registry, &releases, &answer, &request).expect("no-op");

        assert!(matches!(again, Outcome::AlreadyInstalled(_)));
        assert_eq!(releases.requested.borrow().len(), 3);
    }

    #[test]
    fn prune_removes_other_versions_without_asking() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let old_store = store_in(&temp, "v0.1.0");
        let store = store_in(&temp, "v0.2.0");
        let assets = [
            (url("v0.1.0"), pack_zip(WEIGHTS)),
            (url("v0.2.0"), pack_zip(WEIGHTS)),
        ];
        let releases = FakeReleases::with(&assets, "v0.2.0");
        let answer = Answer::of(false);
        let registry = fake_registry();
        let request = args(false, false);
        pull(&old_store, &registry, &releases, &answer, &request).expect("pulled");

        let request = args(true, false);
        let outcome = pull(&store, &registry, &releases, &answer, &request).expect("pulled");

        assert!(matches!(&outcome, Outcome::Installed { pruned, .. } if *pruned == ["v0.1.0"]));
        assert!(!answer.asked.get());
        assert!(!store.pack_dir("v0.1.0", "fake").exists());
    }

    #[test]
    fn without_prune_the_answer_decides() {
        for delete in [false, true] {
            let temp = tempfile::tempdir().expect("a temp dir");
            let old_store = store_in(&temp, "v0.1.0");
            let store = store_in(&temp, "v0.2.0");
            let assets = [
                (url("v0.1.0"), pack_zip(WEIGHTS)),
                (url("v0.2.0"), pack_zip(WEIGHTS)),
            ];
            let releases = FakeReleases::with(&assets, "v0.2.0");
            let answer = Answer::of(delete);
            let registry = fake_registry();
            let request = args(false, false);
            pull(&old_store, &registry, &releases, &answer, &request).expect("pulled");

            pull(&store, &registry, &releases, &answer, &request).expect("pulled");

            let old_kept = store.pack_dir("v0.1.0", "fake").exists();
            assert!(answer.asked.get());
            assert_eq!(old_kept, !delete);
        }
    }

    #[test]
    fn with_no_other_versions_nothing_is_asked() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let store = store_in(&temp, "v0.2.0");
        let assets = [(url("v0.2.0"), pack_zip(WEIGHTS))];
        let releases = FakeReleases::with(&assets, "v0.2.0");
        let answer = Answer::of(true);
        let registry = fake_registry();

        let request = args(false, false);
        pull(&store, &registry, &releases, &answer, &request).expect("pulled");

        assert!(!answer.asked.get());
    }

    #[test]
    fn a_corrupt_download_installs_nothing() {
        let temp = tempfile::tempdir().expect("a temp dir");
        let store = store_in(&temp, "v0.1.0");
        let assets = [(url("v0.1.0"), pack_zip(b"tampered"))];
        let releases = FakeReleases::with(&assets, "v0.1.0");
        let answer = Answer::of(false);
        let registry = fake_registry();

        let request = args(false, false);
        let error = pull(&store, &registry, &releases, &answer, &request).expect_err("refused");

        let temp_path = temp.path();
        let listing = std::fs::read_dir(temp_path).expect("listed");
        assert!(matches!(error, PullError::Install(_)));
        assert_eq!(listing.count(), 0);
    }

    #[test]
    fn only_y_or_yes_accepts() {
        for answer in ["y", "Y\n", " yes ", "YES\r\n"] {
            assert!(accepts(answer), "{answer:?}");
        }

        for answer in ["", "\n", "n", "no", "yep", "maybe"] {
            assert!(!accepts(answer), "{answer:?}");
        }
    }

    #[test]
    fn the_latest_tag_is_read_from_the_release() {
        let tag = latest_tag_from("url", "{\"tag_name\": \"v1.2.3\", \"name\": \"x\"}").expect("read");
        let error = latest_tag_from("url", "{\"name\": \"x\"}").expect_err("no tag");

        assert_eq!(tag, "v1.2.3");
        assert!(error.to_string().contains("url"), "{error}");
    }
}
