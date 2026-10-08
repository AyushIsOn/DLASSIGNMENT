#!/usr/bin/env bash
# Build android/AcharyaGPT.apk without Gradle (Android SDK build-tools + platform only).
#
#   ANDROID_SDK=/path/to/sdk bash android/build_apk.sh
#
# Needs: JDK 17+, $ANDROID_SDK/platforms/android-34, $ANDROID_SDK/build-tools/34.0.0, Python
# with cairosvg + Pillow (icon). Signed with a local debug key (fine for installing on
# your own phone; not for the Play Store).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK="${ANDROID_SDK:?set ANDROID_SDK to the Android SDK folder}"
BT="$SDK/build-tools/34.0.0"
JAR="$SDK/platforms/android-34/android.jar"
OUT="$HERE/build"
PY="${PYTHON:-python3}"
rm -rf "$OUT" && mkdir -p "$OUT/res" "$OUT/classes" "$OUT/dex" "$OUT/assets"
# the chat UI: the same page the API serves at GET /
cp "$HERE/../src/acharya/web/index.html" "$OUT/assets/index.html"

echo "== icon"
"$PY" - "$HERE/../AcharyaGPT(iOS)/AcharyaGPT(iOS)/Assets.xcassets/acharyaLogoLarge.imageset/Layer 1-2.svg" "$OUT/res" <<'EOF'
import io, sys
from pathlib import Path
import cairosvg
from PIL import Image
svg, res = sys.argv[1], Path(sys.argv[2])
for folder, size in {"mipmap-mdpi": 48, "mipmap-hdpi": 72, "mipmap-xhdpi": 96,
                     "mipmap-xxhdpi": 144, "mipmap-xxxhdpi": 192}.items():
    logo = Image.open(io.BytesIO(cairosvg.svg2png(url=svg, output_width=int(size * 0.62))))
    icon = Image.new("RGBA", (size, size), (13, 15, 15, 255))
    icon.paste(logo, ((size - logo.width) // 2, (size - logo.height) // 2), logo.convert("RGBA"))
    (res / folder).mkdir(parents=True, exist_ok=True)
    icon.save(res / folder / "ic_launcher.png")
EOF

echo "== resources + manifest"
"$BT/aapt2" compile --dir "$OUT/res" -o "$OUT/res.zip"
"$BT/aapt2" link -o "$OUT/unsigned.apk" -I "$JAR" --manifest "$HERE/AndroidManifest.xml" \
  --min-sdk-version 24 --target-sdk-version 34 --version-code 2 --version-name 2.0 \
  -A "$OUT/assets" "$OUT/res.zip"

echo "== compile Java -> dex"
javac --release 11 -classpath "$JAR" -d "$OUT/classes" \
  $(find "$HERE/src" -name '*.java') 2>&1 | grep -v "^warning: \[options\]\|^1 warning" || true
[[ -f "$OUT/classes/co/acharyagpt/app/MainActivity.class" ]] || { echo "javac failed"; exit 1; }
"$BT/d8" --lib "$JAR" --min-api 24 --release --output "$OUT/dex" \
  $(find "$OUT/classes" -name '*.class')
(cd "$OUT/dex" && zip -q -j "$OUT/unsigned.apk" classes.dex)

echo "== align + sign"
KEY="$HERE/debug.keystore"
if [[ ! -f "$KEY" ]]; then
  keytool -genkeypair -keystore "$KEY" -storepass android -keypass android -alias debug \
    -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=AcharyaGPT Debug" >/dev/null 2>&1
fi
"$BT/zipalign" -f -p 4 "$OUT/unsigned.apk" "$OUT/aligned.apk"
"$BT/apksigner" sign --ks "$KEY" --ks-pass pass:android --key-pass pass:android \
  --out "$HERE/AcharyaGPT.apk" "$OUT/aligned.apk"
"$BT/apksigner" verify "$HERE/AcharyaGPT.apk"
ls -lh "$HERE/AcharyaGPT.apk"
