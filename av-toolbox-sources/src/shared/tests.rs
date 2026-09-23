/// A 4 by 2 `C420` y4m header with no frames after it.
pub(crate) fn y4m_header_only() -> Vec<u8> {
    let mut bytes = Vec::new();

    ::y4m::encode(4, 2, ::y4m::Ratio::new(25, 1))
        .write_header(&mut bytes)
        .expect("writes the header");

    bytes
}
