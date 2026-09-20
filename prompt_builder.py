// mentor.js - AI Mentor chat (plain JavaScript). Put in static/js/mentor.js
function initMentor() {
  var chat = document.getElementById("mentor-chat");
  var form = document.getElementById("mentor-form");
  var input = document.getElementById("mentor-input");
  var sendBtn = document.getElementById("mentor-send");
  var sugg = document.getElementById("mentor-suggestions");
  if (!chat || !form || !input || !sendBtn) {
    console.error("AI Mentor: page elements not found. Use the latest templates/mentor.html.");
    return;
  }
  var ASK_URL = chat.getAttribute("data-ask-url") || "/mentor/ask";
  var history = [], busy = false;

  function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function inline(s) { return esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>"); }

  function render(text) {
    var lines = text.split("\n"), html = "", i = 0;
    var isRow = function (l) { return /^\s*\|.*\|\s*$/.test(l); };
    var isLi = function (l) { return /^\s*[*-]\s+/.test(l); };
    while (i < lines.length) {
      var line = lines[i];
      if (isRow(line)) {
        var rows = [];
        while (i < lines.length && isRow(lines[i])) rows.push(lines[i++]);
        var cells = function (r) { return r.trim().replace(/^\||\|$/g, "").split("|").map(function (c) { return c.trim(); }); };
        var body = rows.filter(function (r) { return !/^\s*\|[\s:|-]+\|\s*$/.test(r); });
        if (body.length) {
          html += '<div class="overflow-x-auto"><table class="my-2 border-collapse text-xs"><thead><tr>' +
            cells(body[0]).map(function (c) { return '<th class="border border-slate-200 bg-white px-2 py-1 text-left">' + inline(c) + "</th>"; }).join("") +
            "</tr></thead><tbody>" +
            body.slice(1).map(function (r) { return "<tr>" + cells(r).map(function (c) { return '<td class="border border-slate-200 px-2 py-1">' + inline(c) + "</td>"; }).join("") + "</tr>"; }).join("") +
            "</tbody></table></div>";
        }
        continue;
      }
      if (isLi(line)) {
        html += '<ul class="my-1 list-disc pl-5">';
        while (i < lines.length && isLi(lines[i])) html += "<li>" + inline(lines[i++].replace(/^\s*[*-]\s+/, "")) + "</li>";
        html += "</ul>";
        continue;
      }
      if (!line.trim()) { i++; continue; }
      var para = [];
      while (i < lines.length && lines[i].trim() && !isLi(lines[i]) && !isRow(lines[i])) para.push(lines[i++]);
      html += '<p class="my-1">' + para.map(inline).join("<br>") + "</p>";
    }
    return html;
  }

  function add(kind, content, asHtml) {
    var empty = document.getElementById("mentor-empty");
    if (empty) empty.remove();
    var wrap = document.createElement("div");
    var bubble = document.createElement("div");
    if (kind === "user") {
      wrap.className = "flex justify-end";
      bubble.className = "max-w-[85%] rounded-xl bg-blue-600 px-3 py-2 text-white";
    } else if (kind === "error") {
      bubble.className = "rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-red-700";
    } else {
      bubble.className = "rounded-xl border border-slate-200 bg-white px-3 py-2";
    }
    if (asHtml) bubble.innerHTML = content; else bubble.textContent = content;
    wrap.appendChild(bubble);
    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;
    return wrap;
  }

  function csrfHeaders() {
    var m = document.querySelector('meta[name="csrf-token"]');
    return m ? { "X-CSRFToken": m.content } : {};
  }

  function ask(q) {
    q = q.trim();
    if (!q || busy) return;
    add("user", q);
    var pending = add("bot", "Checking your records...");
    busy = true; sendBtn.disabled = true;

    fetch(ASK_URL, {
      method: "POST",
      credentials: "same-origin",
      headers: Object.assign({ "Content-Type": "application/json" }, csrfHeaders()),
      body: JSON.stringify({ question: q, history: history })
    })
    .then(function (res) { return res.json().catch(function () { return {}; }).then(function (d) { return { ok: res.ok, status: res.status, d: d }; }); })
    .then(function (r) {
      pending.remove();
      if (r.ok && r.d.answer) {
        add("bot", render(r.d.answer), true);
        history.push({ role: "user", content: q }, { role: "assistant", content: r.d.answer });
        history = history.slice(-6);
      } else if (r.status === 401) {
        add("error", "Your session has expired. Please log in again.");
      } else {
        add("error", r.d.error || "Something went wrong. Please try again.");
      }
    })
    .catch(function () { pending.remove(); add("error", "Could not reach the server. Check your connection."); })
    .finally(function () { busy = false; sendBtn.disabled = false; input.focus(); });
  }

  form.addEventListener("submit", function (e) { e.preventDefault(); var q = input.value; input.value = ""; ask(q); });
  if (sugg) sugg.addEventListener("click", function (e) {
    var b = e.target.closest("button"); if (b) ask(b.textContent);
  });
  console.log("AI Mentor ready");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initMentor);
} else {
  initMentor();
}
