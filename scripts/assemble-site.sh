#!/usr/bin/env bash
set -euo pipefail

usage='usage: assemble-site.sh UPSTREAM_ROOT PAGES_ROOT DESTINATION [UPSTREAM_REF]'

upstream_root=${1:?$usage}
pages_root=${2:?$usage}
destination=${3:?$usage}
upstream_ref=${4:-}

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

# A reader has to be able to tell which TileFoundry a page describes without
# leaving the page, so the ref being rendered goes into the footer every page
# carries. Assembling without a ref is a local preview and claims nothing.
if [[ -n "$upstream_ref" ]]; then
  if ! grep -q '^copyright: ' "$destination/mkdocs.yml"; then
    printf 'No copyright line in mkdocs.yml to carry the rendered ref\n' >&2
    exit 1
  fi
  sed -i "s|^copyright: |copyright: Documenting TileFoundry ${upstream_ref} \&middot; |" \
    "$destination/mkdocs.yml"
fi
