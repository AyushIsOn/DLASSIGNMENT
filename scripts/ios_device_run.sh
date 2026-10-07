#!/usr/bin/env bash
# Build, install and launch AcharyaGPT on a physical iPhone (macOS + Xcode, cable connected).
#
#   bash scripts/ios_device_run.sh                # first connected physical iPhone
#   TEAM=HM8NC475WC UDID=<id> bash scripts/ios_device_run.sh
#
# On failure it prints the real Xcode error lines and what to do about them.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PROJECT="AcharyaGPT(iOS)/AcharyaGPT(iOS).xcodeproj"
SCHEME="AcharyaGPT(iOS)"
TEAM="${TEAM:-HM8NC475WC}"
BUNDLE="${BUNDLE:-com.ayushgupta.AcharyaGPT}"
LOG="$ROOT/build/ios_build.log"
mkdir -p build

say() { echo; echo "== $*"; }
fail() { echo; echo "FAILED: $*"; exit 1; }

say "1/5 Xcode"
DEV_DIR="$(xcode-select -p 2>/dev/null)"
echo "developer dir: $DEV_DIR"
if [[ "$DEV_DIR" == *CommandLineTools* ]]; then
  fail "the command line tools are selected instead of Xcode. Run:
    sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
  then run this script again."
fi
xcodebuild -version || fail "xcodebuild not working - open Xcode once and accept the licence: sudo xcodebuild -license accept"
xcodebuild -runFirstLaunch >/dev/null 2>&1 || true

say "2/5 iPhone"
if [[ -z "${UDID:-}" ]]; then
  UDID="$(xcrun devicectl list devices 2>/dev/null | awk '/physical/ && /connected/ {for (i=1;i<=NF;i++) if ($i ~ /^[0-9A-F]{8}-[0-9A-F]{16}$/) print $i}' | head -1)"
fi
[[ -n "$UDID" ]] || fail "no connected physical iPhone. Plug it in, unlock it, tap 'Trust This Computer'."
echo "device: $UDID"

say "3/5 Can Xcode build for this iPhone?"
DEST="$(xcodebuild -project "$PROJECT" -scheme "$SCHEME" -showdestinations 2>&1)"
if ! grep -q "$UDID" <<<"$DEST"; then
  echo "$DEST" | grep -i "iphone\|error\|ineligible" | head -15
  echo
  echo "The iPhone is not an eligible destination. Usual causes and fixes:"
  echo " - iOS platform/device support missing  -> xcodebuild -downloadPlatform iOS   (several GB, once)"
  echo " - Developer Mode off                   -> iPhone: Settings > Privacy & Security > Developer Mode > On"
  echo " - Xcode still preparing the phone      -> open Xcode > Window > Devices and Simulators, wait until"
  echo "                                           the iPhone shows no 'preparing' / 'copying symbols' status"
  fail "fix the above, then run this script again"
fi
echo "OK"

say "4/5 Build (signing team $TEAM, bundle id $BUNDLE)"
if ! xcodebuild -project "$PROJECT" -scheme "$SCHEME" -configuration Debug \
     -destination "id=$UDID" -derivedDataPath build/DerivedData \
     -allowProvisioningUpdates -allowProvisioningDeviceRegistration \
     DEVELOPMENT_TEAM="$TEAM" PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE" CODE_SIGN_STYLE=Automatic \
     build > "$LOG" 2>&1; then
  echo "---- Xcode errors (full log: $LOG) ----"
  grep -E "error:|\*\* BUILD FAILED" "$LOG" | sort -u | head -20
  echo "---------------------------------------"
  if grep -qiE "No Accounts|No account for team|not signed in" "$LOG"; then
    echo "FIX: Xcode > Settings > Accounts > + > Apple ID: sign in with ayushgupta.2406@gmail.com"
  fi
  if grep -qiE "No profiles for|provisioning profile|Signing for .* requires a development team" "$LOG"; then
    echo "FIX: open the project in Xcode once:  open \"$PROJECT\""
    echo "     target AcharyaGPT(iOS) > Signing & Capabilities > tick 'Automatically manage signing',"
    echo "     Team = Ayush Gupta (Personal Team), Bundle Identifier = $BUNDLE. Then run this again."
  fi
  if grep -qiE "is not available|cannot be used|not installed|Developer Mode" "$LOG"; then
    echo "FIX: xcodebuild -downloadPlatform iOS   and enable Developer Mode on the iPhone"
  fi
  if grep -qiE "already in use|not available.*bundle identifier|cannot be registered" "$LOG"; then
    echo "FIX: pick another bundle id:  BUNDLE=com.ayushgupta.acharya2 bash scripts/ios_device_run.sh"
  fi
  fail "build failed - send the error lines above"
fi
APP="build/DerivedData/Build/Products/Debug-iphoneos/AcharyaGPT(iOS).app"
[[ -d "$APP" ]] || fail "build said OK but $APP is missing"
echo "BUILD SUCCEEDED"

say "5/5 Install + launch"
xcrun devicectl device install app --device "$UDID" "$APP" || fail "install failed (is the iPhone unlocked?)"
if ! xcrun devicectl device process launch --device "$UDID" "$BUNDLE"; then
  echo
  echo "Installed, but iOS blocked the launch. On the iPhone:"
  echo "  Settings > General > VPN & Device Management > your Apple ID > Trust"
  echo "then tap the AcharyaGPT icon (or run this script again)."
fi
echo
echo "Next: start the server and paste its URL into the app (gear icon):"
echo "  bash scripts/mac_serve.sh --lan      # Mac on the same Wi-Fi -> prints http://<mac-ip>:8000"
