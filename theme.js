// Apply saved theme immediately to prevent flash of wrong theme
(function () {
  var saved = localStorage.getItem('ccm-theme');
  if (saved === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
})();

document.addEventListener('DOMContentLoaded', function () {
  var btn = document.getElementById('theme-toggle');
  if (!btn) return;

  btn.addEventListener('click', function () {
    var isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    if (isDark) {
      document.documentElement.removeAttribute('data-theme');
      localStorage.setItem('ccm-theme', 'light');
      btn.setAttribute('aria-label', 'Switch to dark mode');
    } else {
      document.documentElement.setAttribute('data-theme', 'dark');
      localStorage.setItem('ccm-theme', 'dark');
      btn.setAttribute('aria-label', 'Switch to light mode');
    }
  });

  // Set initial aria-label
  var isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  btn.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
});
