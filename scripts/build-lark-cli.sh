#!/usr/bin/env bash
# Compile the two CLIs independently, then atomically publish into a bind mount.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:-${project_root}/docker_data/lark-cli}"
mkdir -p -- "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"
build_dir="$(mktemp -d "${output_dir}/.build-XXXXXXXX")"
trap 'rm -rf -- "$build_dir"' EXIT

case "$(uname -m)" in
    x86_64) target_arch=amd64 ;;
    aarch64|arm64) target_arch=arm64 ;;
    *) printf 'Unsupported host architecture\n' >&2; exit 1 ;;
esac

docker buildx build \
    --file "${project_root}/Dockerfile" \
    --target lark-cli-export \
    --build-arg "LARK_CLI_REF=${LARK_CLI_REF:-v1.0.89}" \
    --platform "${LARK_CLI_PLATFORM:-linux/${target_arch}}" \
    --output "type=local,dest=${build_dir}" \
    "$project_root"

# Publish atomically so running processes retain their old executable.
# File bind mounts retain the old inode: recreate consumers after updating.
for binary in lark-cli lark-cli-auth; do
    chmod 0755 "${build_dir}/${binary}"
    mv -f -- "${build_dir}/${binary}" "${output_dir}/${binary}"
done
printf 'Published lark-cli and lark-cli-auth to %s\n' "$output_dir"
printf 'Recreate backend containers to refresh file mounts; existing sandboxes refresh after recycling.\n'
