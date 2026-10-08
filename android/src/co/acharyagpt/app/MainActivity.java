package co.acharyagpt.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.os.Bundle;
import android.text.InputType;
import android.view.Gravity;
import android.view.ViewGroup;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;

/**
 * AcharyaGPT for Android: shows the server's web chat (GET /) in a WebView.
 * The server URL (https://....trycloudflare.com from lightning_serve.sh, or
 * http://<Mac-IP>:8000 from mac_serve.sh --lan) is set with the "Server" button.
 */
public class MainActivity extends Activity {
    private static final String PREFS = "acharya";
    private static final String KEY_URL = "backendURL";
    private WebView web;
    private TextView status;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        int dark = Color.rgb(13, 15, 15);
        getWindow().setStatusBarColor(dark);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(dark);

        LinearLayout bar = new LinearLayout(this);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setPadding(dp(16), dp(8), dp(8), dp(8));
        TextView title = new TextView(this);
        title.setText("AcharyaGPT");
        title.setTextColor(Color.WHITE);
        title.setTextSize(18);
        bar.addView(title, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        Button server = new Button(this);
        server.setText("Server");
        server.setOnClickListener(v -> askForUrl());
        bar.addView(server);
        root.addView(bar);

        status = new TextView(this);
        status.setTextColor(Color.rgb(248, 113, 113));
        status.setPadding(dp(16), 0, dp(16), dp(4));
        root.addView(status);

        web = new WebView(this);
        web.setBackgroundColor(dark);
        WebSettings settings = web.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        web.setWebViewClient(new WebViewClient() {
            @Override
            public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
                if (request.isForMainFrame()) {
                    status.setText("Cannot reach the server (" + error.getDescription()
                            + "). Is it running? Tap Server to change the URL.");
                }
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                if (!url.startsWith("about:")) status.setText("");
            }
        });
        root.addView(web, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));
        setContentView(root);

        String url = prefs().getString(KEY_URL, "");
        if (url.isEmpty()) askForUrl(); else load(url);
    }

    private SharedPreferences prefs() {
        return getSharedPreferences(PREFS, MODE_PRIVATE);
    }

    private void load(String url) {
        status.setText("Connecting to " + url + " ...");
        web.loadUrl(url + "/");
    }

    private void askForUrl() {
        EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        input.setHint("https://....trycloudflare.com");
        input.setText(prefs().getString(KEY_URL, ""));
        new AlertDialog.Builder(this)
                .setTitle("AcharyaGPT server")
                .setMessage("Lightning GPU: the https URL printed by lightning_serve.sh\n"
                        + "Mac on the same Wi-Fi: http://<Mac-IP>:8000")
                .setView(input)
                .setPositiveButton("Connect", (d, w) -> {
                    String url = normalize(input.getText().toString());
                    prefs().edit().putString(KEY_URL, url).apply();
                    load(url);
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    static String normalize(String raw) {
        String value = raw.trim();
        while (value.endsWith("/")) value = value.substring(0, value.length() - 1);
        if (!value.startsWith("http://") && !value.startsWith("https://")) {
            String host = value.split("[:/]")[0];
            boolean local = host.matches("\\d+\\.\\d+\\.\\d+\\.\\d+") || host.equals("localhost")
                    || host.endsWith(".local");
            value = (local ? "http://" : "https://") + value;
        }
        return value;
    }

    @Override
    public void onBackPressed() {
        if (web.canGoBack()) web.goBack(); else super.onBackPressed();
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
