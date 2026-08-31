#!/usr/bin/env bash
# Local preview: assemble upstream docs + this repo's overlay, then run mkdocs serve.
# Unlike scripts/assemble-site.sh (used by CI, which copies), this builds a symlink
# farm so edits in either source tree show up on live reload.
set -euo pipefail

pages_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
upstream_root=$(cd "${TILEFOUNDRY_SRC:-$pages_root/../TileFoundry}" && pwd)
destination="$pages_root/_site/source"

if [[ ! -d "$upstream_root/docs" ]]; then
  printf 'Missing upstream documentation directory: %s\n' "$upstream_root/docs" >&2
  printf 'Set TILEFOUNDRY_SRC to the TileFoundry checkout.\n' >&2
  exit 1
fi

rm -rf "$destination"
mkdir -p "$destination/docs"
cp -asf "$upstream_root/docs/." "$destination/docs/"
cp -asf "$pages_root/site/docs/." "$destination/docs/"
ln -sf "$pages_root/site/mkdocs.yml" "$destination/mkdocs.yml"

exec mkdocs serve \
  --config-file "$destination/mkdocs.yml" \
  --watch "$upstream_root/docs" \
  --watch "$pages_root/site" \
  "$@"
