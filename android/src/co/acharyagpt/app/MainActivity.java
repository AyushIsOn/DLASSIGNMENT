package co.acharyagpt.app;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.view.WindowManager;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

/**
 * AcharyaGPT for Android. The whole chat UI (src/acharya/web/index.html) is bundled in the
 * APK and talks to the AcharyaGPT server set in its Settings sheet, so the app opens
 * instantly and the design does not depend on the server.
 */
public class MainActivity extends Activity {
    private WebView web;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        int dark = Color.rgb(11, 13, 13);
        getWindow().setStatusBarColor(dark);
        getWindow().setNavigationBarColor(dark);
        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);

        web = new WebView(this);
        web.setBackgroundColor(dark);
        WebSettings settings = web.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);           // saved server URL + chat history
        settings.setAllowFileAccess(true);             // file:///android_asset/
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);  // http://<Mac-IP>
        settings.setTextZoom(100);
        web.setWebViewClient(new WebViewClient());
        web.setOverScrollMode(WebView.OVER_SCROLL_NEVER);
        setContentView(web);
        if (state != null) web.restoreState(state); else web.loadUrl("file:///android_asset/index.html");
    }

    @Override
    protected void onSaveInstanceState(Bundle out) {
        super.onSaveInstanceState(out);
        web.saveState(out);
    }

    @Override
    public void onBackPressed() {
        // close the settings sheet first, then leave the app
        web.evaluateJavascript(
                "(function(){var s=document.getElementById('sheet');"
                        + "if(s&&s.classList.contains('open')){closeSheet();return 1}return 0})()",
                value -> {
                    if (!"1".equals(value)) MainActivity.super.onBackPressed();
                });
    }
}
