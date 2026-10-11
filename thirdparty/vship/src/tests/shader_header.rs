use std::fs;

use crate::shader_header::write_shader_header;

fn header_for(files: &[(&str, &[u8])]) -> String {
    let folder = tempfile::tempdir().expect("a temp dir");
    let shader_dir = folder.path().join("shaders");
    fs::create_dir(&shader_dir).expect("creates the shader dir");

    for (name, bytes) in files {
        let path = shader_dir.join(name);
        fs::write(path, bytes).expect("writes a shader");
    }

    let output = folder.path().join("shaders.hpp");
    write_shader_header(&shader_dir, &output).expect("writes the header");

    fs::read_to_string(&output).expect("reads the header")
}

#[test]
fn the_header_holds_every_shader_with_its_bytes() {
    let header = header_for(&[("a.spv", &[0x01, 0xff]), ("b.spv", &[0x80])]);

    assert!(header.contains("{\"a.spv\", {1,-1,}}"));
    assert!(header.contains("{\"b.spv\", {-128,}}"));
}

#[test]
fn gitkeep_is_skipped() {
    let header = header_for(&[(".gitkeep", &[]), ("a.spv", &[0x01])]);

    assert!(!header.contains(".gitkeep"));
    assert!(header.contains("{\"a.spv\", {1,}}"));
}
