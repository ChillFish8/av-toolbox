use std::fs;
use std::io::{self, Write};
use std::path::Path;

/// Writes every shader in `shader_dir` into a C++ header as Vship's own embedder does.
///
/// Each file becomes one `{"<file name>", {<signed byte values>}}` entry of
/// `shaderNameToByteCode`, in file name order so the output is reproducible.
pub fn write_shader_header(shader_dir: &Path, output: &Path) -> io::Result<()> {
    let mut names = Vec::new();
    for entry in fs::read_dir(shader_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name != ".gitkeep" {
            names.push(name);
        }
    }

    names.sort();

    let mut header = String::new();
    header.push_str("#pragma once\n#include<string>\n#include<unordered_map>\n#include<vector>\n\n");
    header.push_str("std::unordered_map<std::string, std::vector<char>> shaderNameToByteCode = {\n");

    for name in &names {
        let path = shader_dir.join(name);
        let bytecode = fs::read(path)?;

        header.push_str("{\"");
        header.push_str(name);
        header.push_str("\", {");

        for byte in bytecode {
            let signed = i8::from_ne_bytes([byte]);
            header.push_str(&signed.to_string());
            header.push(',');
        }

        header.push_str("}},\n");
    }

    header.push_str("};\n");

    let mut file = fs::File::create(output)?;
    file.write_all(header.as_bytes())
}
