#!/usr/bin/env bash
set -euo pipefail

upstream_root=${1:?usage: assemble-site.sh UPSTREAM_ROOT PAGES_ROOT DESTINATION}
pages_root=${2:?usage: assemble-site.sh UPSTREAM_ROOT PAGES_ROOT DESTINATION}
destination=${3:?usage: assemble-site.sh UPSTREAM_ROOT PAGES_ROOT DESTINATION}

if [[ ! -d "$upstream_root/docs" ]]; then
  printf 'Missing upstream documentation directory: %s\n' "$upstream_root/docs" >&2
  exit 1
fi

if [[ ! -f "$pages_root/mkdocs.yml" || ! -d "$pages_root/docs" ]]; then
  printf 'Missing Pages site overlay in: %s\n' "$pages_root" >&2
  exit 1
fi

if [[ -e "$destination" && -n "$(find "$destination" -mindepth 1 -print -quit)" ]]; then
  printf 'Destination must be empty: %s\n' "$destination" >&2
  exit 1
fi

mkdir -p "$destination/docs"
cp -a "$upstream_root/docs/." "$destination/docs/"
cp -a "$pages_root/docs/." "$destination/docs/"
cp -a "$pages_root/mkdocs.yml" "$destination/mkdocs.yml"

# Build hooks are part of the site definition; mkdocs resolves the `hooks`
# paths against the directory holding mkdocs.yml.
if [[ -d "$pages_root/hooks" ]]; then
  cp -a "$pages_root/hooks" "$destination/hooks"
fi
