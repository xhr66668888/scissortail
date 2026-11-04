#!/usr/bin/env bash
# Builds the Scissortail installers on a Mac with Apple silicon, into dist/:
#
#   scripts/build.sh                 all four
#   scripts/build.sh macos-arm64     Scissortail_<v>_macOS-AppleSilicon.dmg   macOS 13.4 or later
#   scripts/build.sh macos-x64       Scissortail_<v>_macOS-Intel.dmg          macOS 10.15 or later
#   scripts/build.sh windows-x64     Scissortail_<v>_Windows-x64-setup.exe    Windows 7 SP1 – 11
#   scripts/build.sh windows-arm64   Scissortail_<v>_Windows-ARM64-setup.exe  Windows 11 on Arm
#
# The Windows installers are cross-compiled. The tools for that are fetched
# once into $SCISSORTAIL_TOOLS (default ~/Library/Caches/scissortail-build):
# LLVM (clang-cl, lld-link, llvm-rc), cargo-xwin — which downloads Microsoft's
# C runtime and Windows SDK libraries; building accepts Microsoft's license
# for them — NSIS (makensis) and YY-Thunks, which lets the x64 build start on
# Windows 7.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="${SCISSORTAIL_TOOLS:-$HOME/Library/Caches/scissortail-build}"
DIST="$ROOT/dist"
VERSION="$(sed -n 's/^  "version": "\(.*\)",$/\1/p' "$ROOT/app/src-tauri/tauri.conf.json")"

LLVM_VERSION=23.1.4
LLVM_SHA256=5d3179db39adeca5b17a34d82708fe976bc60e055b05fe557878fb3fb64d39ef
CARGO_XWIN_VERSION=0.23.1
MAKENSIS_VERSION=3.13
MAKENSIS_SHA256=fb902fdc6f21f2d0dee4f78d6a1350263901624217ed2ae591b94d107e665a38 # Homebrew bottle, arm64_golden_gate
YY_THUNKS_VERSION=1.2.3
YY_THUNKS_SHA256=4dc68ec8784e0afed7284e4393aec672873608161c3a5ae52faf618875d95b81

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# fetch URL SHA256 FILE [curl options]: download once and check the checksum.
fetch() {
  local url=$1 sha=$2 out=$3
  shift 3
  [[ -f $out ]] && return 0
  mkdir -p "$(dirname "$out")"
  curl -fL --retry 3 "$@" -o "$out.part" "$url"
  if ! echo "$sha  $out.part" | shasum -a 256 -c - >/dev/null; then
    rm -f "$out.part"
    echo "checksum mismatch for $url" >&2
    exit 1
  fi
  mv "$out.part" "$out"
}

tauri_build() { (cd "$ROOT/app/src-tauri" && cargo tauri build "$@"); }

publish() {
  mkdir -p "$DIST"
  cp -f "$1" "$DIST/$2"
  echo "-> dist/$2"
}

# drop_app TRIPLE: remove the .app the DMG was made from, so that Spotlight
# does not list it next to the copy installed in /Applications.
drop_app() { rm -rf "$ROOT/target/$1/release/bundle/macos/Scissortail.app"; }

build_macos_arm64() {
  say "macOS, Apple silicon"
  rustup target add aarch64-apple-darwin >/dev/null
  tauri_build --target aarch64-apple-darwin --bundles app dmg
  publish "$ROOT/target/aarch64-apple-darwin/release/bundle/dmg/Scissortail_${VERSION}_aarch64.dmg" "Scissortail_${VERSION}_macOS-AppleSilicon.dmg"
  drop_app aarch64-apple-darwin
}

build_macos_x64() {
  say "macOS, Intel"
  rustup target add x86_64-apple-darwin >/dev/null
  MACOSX_DEPLOYMENT_TARGET=10.15 tauri_build --target x86_64-apple-darwin --bundles app dmg \
    --config '{"bundle":{"macOS":{"minimumSystemVersion":"10.15"}}}'
  publish "$ROOT/target/x86_64-apple-darwin/release/bundle/dmg/Scissortail_${VERSION}_x64.dmg" "Scissortail_${VERSION}_macOS-Intel.dmg"
  drop_app x86_64-apple-darwin
}

windows_tools() {
  local dl="$TOOLS/downloads"
  if [[ ! -x $TOOLS/llvm/bin/clang ]]; then
    say "LLVM $LLVM_VERSION"
    fetch "https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/LLVM-$LLVM_VERSION-macOS-ARM64.tar.xz" \
      "$LLVM_SHA256" "$dl/LLVM-$LLVM_VERSION-macOS-ARM64.tar.xz"
    rm -rf "$TOOLS/llvm.tmp" && mkdir -p "$TOOLS/llvm.tmp"
    # only what cross-compiling needs (the full bin/ is 5.5 GB)
    tar -xJf "$dl/LLVM-$LLVM_VERSION-macOS-ARM64.tar.xz" -C "$TOOLS/llvm.tmp" --strip-components 1 \
      --include '*/bin/clang' --include '*/bin/clang-[0-9]*' --include '*/bin/clang-cl' --include '*/bin/clang++' \
      --include '*/bin/lld' --include '*/bin/lld-link' --include '*/bin/llvm-ar' --include '*/bin/llvm-lib' \
      --include '*/bin/llvm-dlltool' --include '*/bin/llvm-rc' --include '*/bin/llvm-readobj' \
      --include '*/lib/clang/*/include/*'
    mv "$TOOLS/llvm.tmp" "$TOOLS/llvm"
    rm -f "$dl/LLVM-$LLVM_VERSION-macOS-ARM64.tar.xz"
  fi
  if [[ ! -x $TOOLS/cargo-xwin/bin/cargo-xwin ]]; then
    say "cargo-xwin $CARGO_XWIN_VERSION"
    cargo install --locked cargo-xwin --version "$CARGO_XWIN_VERSION" --root "$TOOLS/cargo-xwin"
  fi
  if [[ ! -x $TOOLS/makensis/bin/makensis ]]; then
    say "NSIS $MAKENSIS_VERSION"
    fetch "https://ghcr.io/v2/homebrew/core/makensis/blobs/sha256:$MAKENSIS_SHA256" \
      "$MAKENSIS_SHA256" "$dl/makensis-$MAKENSIS_VERSION.bottle.tar.gz" -H "Authorization: Bearer QQ=="
    rm -rf "$TOOLS/makensis.tmp" && mkdir -p "$TOOLS/makensis.tmp"
    tar -xzf "$dl/makensis-$MAKENSIS_VERSION.bottle.tar.gz" -C "$TOOLS/makensis.tmp" --strip-components 2
    mv "$TOOLS/makensis.tmp" "$TOOLS/makensis"
  fi
  if [[ ! -f $TOOLS/yy-thunks/objs/x64/YY_Thunks_for_Win7.obj ]]; then
    say "YY-Thunks $YY_THUNKS_VERSION"
    fetch "https://github.com/Chuyu-Team/YY-Thunks/releases/download/v$YY_THUNKS_VERSION/YY-Thunks-Objs.zip" \
      "$YY_THUNKS_SHA256" "$dl/YY-Thunks-$YY_THUNKS_VERSION-Objs.zip"
    rm -rf "$TOOLS/yy-thunks" && mkdir -p "$TOOLS/yy-thunks"
    unzip -q "$dl/YY-Thunks-$YY_THUNKS_VERSION-Objs.zip" -d "$TOOLS/yy-thunks"
  fi
  # tauri runs `makensis` from PATH and clears NSISDIR, which tells the
  # relocated Homebrew build where its stubs and plug-ins are; without a UTF-8
  # locale makensis aborts with std::bad_alloc
  mkdir -p "$TOOLS/bin"
  printf '#!/bin/sh\nNSISDIR="%s" LC_ALL=en_US.UTF-8 exec "%s" "$@"\n' "$TOOLS/makensis/share/nsis" "$TOOLS/makensis/bin/makensis" >"$TOOLS/bin/makensis"
  # cargo-xwin links with the `lld-link` found first on PATH. This wrapper
  # puts the files listed in $SCISSORTAIL_LINK_FIRST (':'-separated) before all
  # other inputs — after the "-flavor link" that lld only accepts as its first
  # arguments — so that their definitions win over the import libraries
  # (see build_windows).
  cat >"$TOOLS/bin/lld-link" <<EOF
#!/bin/sh
if [ -n "\$SCISSORTAIL_LINK_FIRST" ]; then
  flavor=
  if [ "\$1" = -flavor ]; then
    flavor=\$2
    shift 2
  fi
  set -f
  IFS=:
  set -- \$SCISSORTAIL_LINK_FIRST "\$@"
  unset IFS
  [ -n "\$flavor" ] && set -- -flavor "\$flavor" "\$@"
fi
exec "$TOOLS/llvm/bin/lld-link" "\$@"
EOF
  # tract-linalg compiles its ARM64 assembly with the GNU-style `clang` driver,
  # but cargo-xwin's C flags are clang-cl style: turn `/imsvc DIR` into
  # `-isystem DIR`.
  cat >"$TOOLS/bin/clang" <<EOF
#!/bin/sh
n=\$#
while [ "\$n" -gt 0 ]; do
  a=\$1
  shift
  n=\$((n - 1))
  if [ "\$a" = /imsvc ] && [ "\$n" -gt 0 ]; then
    set -- "\$@" -isystem "\$1"
    shift
    n=\$((n - 1))
  else
    set -- "\$@" "\$a"
  fi
done
exec "$TOOLS/llvm/bin/clang" "\$@"
EOF
  chmod +x "$TOOLS/bin/makensis" "$TOOLS/bin/lld-link" "$TOOLS/bin/clang"
  export PATH="$TOOLS/bin:$TOOLS/llvm/bin:$TOOLS/cargo-xwin/bin:$PATH"
}

build_windows() {
  local arch=$1 triple label
  case $arch in
    x64) triple=x86_64-pc-windows-msvc label=Windows-x64 ;;
    arm64) triple=aarch64-pc-windows-msvc label=Windows-ARM64 ;;
  esac
  windows_tools
  say "$label"
  rustup target add "$triple" >/dev/null
  if [[ $arch == x64 ]]; then
    # Windows 7 support. Rust's standard library and the static C runtime call
    # a few Windows 8/10 functions, which YY-Thunks supplies on older systems;
    # windows-core imports CoTaskMemAlloc, CoTaskMemFree and
    # CoCreateFreeThreadedMarshaler from combase.dll (Windows 8+), while
    # ole32.dll exports them on every version. Both must be linked before the
    # import libraries (YY-Thunks reports duplicate symbols otherwise), which
    # link arguments cannot do — hence the lld-link wrapper.
    printf 'LIBRARY ole32.dll\nEXPORTS\nCoCreateFreeThreadedMarshaler\nCoTaskMemAlloc\nCoTaskMemFree\n' >"$TOOLS/bin/ole32-win7.def"
    "$TOOLS/llvm/bin/llvm-dlltool" -m i386:x86-64 -d "$TOOLS/bin/ole32-win7.def" -l "$TOOLS/bin/ole32-win7.lib"
    export SCISSORTAIL_LINK_FIRST="$TOOLS/yy-thunks/objs/x64/YY_Thunks_for_Win7.obj:$TOOLS/bin/ole32-win7.lib"
  else
    unset SCISSORTAIL_LINK_FIRST
  fi
  tauri_build --runner cargo-xwin --target "$triple"
  if [[ $arch == x64 ]]; then
    python3 -I "$ROOT/scripts/check_win7_imports.py" "$TOOLS/llvm/bin/llvm-readobj" \
      "$ROOT/target/$triple/release/scissortail-app.exe" "$TOOLS/yy-thunks/Config/x64/6.1.7600.txt"
  fi
  publish "$ROOT/target/$triple/release/bundle/nsis/Scissortail_${VERSION}_${arch}-setup.exe" "Scissortail_${VERSION}_${label}-setup.exe"
}

targets=("$@")
[[ ${#targets[@]} -eq 0 || ${targets[0]} == all ]] && targets=(macos-arm64 macos-x64 windows-x64 windows-arm64)
for t in "${targets[@]}"; do
  case $t in
    macos-arm64) build_macos_arm64 ;;
    macos-x64) build_macos_x64 ;;
    windows-x64) build_windows x64 ;;
    windows-arm64) build_windows arm64 ;;
    *)
      echo "unknown target: $t (macos-arm64, macos-x64, windows-x64, windows-arm64 or all)" >&2
      exit 2
      ;;
  esac
done
# the release carries the alignment model too, for building from source (a
# hard link: no second copy on disk)
ln -f "$ROOT/models/mms-fa-int8.onnx" "$DIST/mms-fa-int8.onnx"
(cd "$DIST" && shasum -a 256 Scissortail_"${VERSION}"_* mms-fa-int8.onnx >SHA256SUMS.txt)
echo
ls -l "$DIST"
