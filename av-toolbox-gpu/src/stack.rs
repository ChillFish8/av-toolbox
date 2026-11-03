use std::ffi::OsStr;

/// Stack bytes the kernel codegen thread needs.
pub const CODEGEN_STACK_BYTES: usize = 16 << 20;

/// Raises `RUST_MIN_STACK` to [CODEGEN_STACK_BYTES] when it is unset.
///
/// Call this at the top of `main`, before any thread is spawned. A thread's stack size is fixed
/// when it starts, so setting the variable later does not reach the threads CubeCL has already
/// made. A value that is already set is kept, even when it is too small. Check it with
/// [codegen_stack_is_sufficient] once logging has started.
///
/// # Safety
///
/// The same rules as [std::env::set_var] apply.
pub unsafe fn raise_codegen_stack_limit() {
    if std::env::var_os("RUST_MIN_STACK").is_some() {
        return;
    }

    // SAFETY: forwarded from this function's own precondition.
    unsafe { std::env::set_var("RUST_MIN_STACK", CODEGEN_STACK_BYTES.to_string()) };
}

/// Whether this process's `RUST_MIN_STACK` is large enough for codegen.
pub fn codegen_stack_is_sufficient() -> bool {
    limit_is_sufficient(std::env::var_os("RUST_MIN_STACK").as_deref())
}

/// Checks a raw `RUST_MIN_STACK` value against [CODEGEN_STACK_BYTES].
///
/// Unset is insufficient, and so is a value that does not parse, because the runtime will not
/// honour one either.
fn limit_is_sufficient(raw: Option<&OsStr>) -> bool {
    raw.and_then(|value| value.to_str())
        .and_then(|value| value.parse::<usize>().ok())
        .is_some_and(|bytes| bytes >= CODEGEN_STACK_BYTES)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::*;

    #[test]
    fn an_unset_limit_is_insufficient() {
        assert!(!limit_is_sufficient(None));
    }

    #[test]
    fn a_limit_below_the_requirement_is_insufficient() {
        for value in ["0", "1024"] {
            let raw = OsString::from(value);
            assert!(!limit_is_sufficient(Some(&raw)), "{value}");
        }

        let just_under = OsString::from((CODEGEN_STACK_BYTES - 1).to_string());
        assert!(!limit_is_sufficient(Some(&just_under)));
    }

    #[test]
    fn the_requirement_itself_is_sufficient() {
        let exact = OsString::from(CODEGEN_STACK_BYTES.to_string());
        assert!(limit_is_sufficient(Some(&exact)));
    }

    #[test]
    fn a_larger_limit_is_sufficient() {
        let larger = OsString::from((CODEGEN_STACK_BYTES * 2).to_string());
        assert!(limit_is_sufficient(Some(&larger)));
    }

    /// Rust does not trim this variable, so neither does the check. A padded value is one the
    /// runtime will not honour either.
    #[test]
    fn a_value_that_does_not_parse_is_insufficient() {
        for value in ["not-a-number", "  16777216  ", "16777216x"] {
            let raw = OsString::from(value);
            assert!(!limit_is_sufficient(Some(&raw)), "{value}");
        }
    }
}
