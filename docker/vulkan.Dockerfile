# Bump ARCH_BUILD_IMAGE, ARCH_RUNTIME_IMAGE and ARCH_SNAPSHOT together when refreshing the pin.
ARG ARCH_BUILD_IMAGE=archlinux:base-devel-20260816.0.574111@sha256:714acd1eef9ae997d95691b1c5220ada0076185b77857c1813f02de0fa83cf7b
ARG ARCH_RUNTIME_IMAGE=archlinux:base-20260816.0.574111@sha256:4bf33b21a715aac0b48ce6e9eaed4782a898eae96f88f5da3635572129c2584a
ARG ARCH_SNAPSHOT=2026/08/17

FROM ${ARCH_BUILD_IMAGE} AS builder

ARG ARCH_SNAPSHOT

RUN printf 'Server = https://archive.archlinux.org/repos/%s/$repo/os/$arch\n' "${ARCH_SNAPSHOT}" > /etc/pacman.d/mirrorlist \
    && pacman -Syyuu --noconfirm --needed --disable-download-timeout \
        git rust clang nasm ffms2 \
    && rm -rf /var/cache/pacman/pkg/*

WORKDIR /build

COPY . .

RUN cargo build --release -p av-toolbox --features vulkan

FROM ${ARCH_RUNTIME_IMAGE} AS runtime

ARG ARCH_SNAPSHOT

RUN printf 'Server = https://archive.archlinux.org/repos/%s/$repo/os/$arch\n' "${ARCH_SNAPSHOT}" > /etc/pacman.d/mirrorlist \
    && pacman -Syyuu --noconfirm --needed --disable-download-timeout \
        ffmpeg ffms2 gcc-libs vulkan-icd-loader vulkan-radeon vulkan-intel \
    && rm -rf /var/cache/pacman/pkg/*

ENV AVTB_MODELS=/models
VOLUME /models

WORKDIR /app

COPY --from=builder /build/target/release/av-toolbox /app/av-toolbox

# Vulkan needs host-provided devices and driver files, so this only proves the binary's own libraries resolve.
RUN /app/av-toolbox --help > /dev/null

ENTRYPOINT ["/app/av-toolbox"]
