/*
 * Rotating open-source thank-you strip (see overrides/main.html).
 *
 * Picks one project from assets/credits.json per page load and writes a
 * gratitude line into #oss-thanks-body. If anything fails, the static
 * fallback text in the template is left untouched.
 *
 * The data URL is resolved relative to THIS script so it works under any site
 * base path (e.g. the GitHub Pages project subpath /sadda/). Rotation re-runs
 * on Material's instant-navigation page changes via the document$ observable.
 */
(function () {
  "use strict";

  var thisScript = document.currentScript;

  function dataUrl() {
    // Resolve ../credits.json relative to assets/js/oss-thanks.js.
    return new URL("../credits.json", thisScript.src).href;
  }

  function render() {
    var el = document.getElementById("oss-thanks-body");
    if (!el) return;
    fetch(dataUrl())
      .then(function (r) {
        return r.ok ? r.json() : Promise.reject(r.status);
      })
      .then(function (items) {
        if (!Array.isArray(items) || items.length === 0) return;
        var pick = items[Math.floor(Math.random() * items.length)];
        if (!pick || !pick.name || !pick.url) return;

        var link = document.createElement("a");
        link.href = pick.url;
        link.rel = "noopener";
        link.textContent = pick.name;

        el.textContent = "sadda uses ";
        el.appendChild(link);
        el.appendChild(
          document.createTextNode(
            " — " +
              pick.used_for +
              ". Thank you to its maintainers and community." +
              (pick.license ? " (" + pick.license + ")" : "")
          )
        );
      })
      .catch(function () {
        /* keep the static fallback line */
      });
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    // Material instant navigation: fires on first load and each page change.
    window.document$.subscribe(render);
  } else if (document.readyState !== "loading") {
    render();
  } else {
    document.addEventListener("DOMContentLoaded", render);
  }
})();
