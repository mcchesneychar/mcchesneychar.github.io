// Apply saved theme immediately to prevent flash of wrong theme
(function () {
  var saved = localStorage.getItem('ccm-theme');
  if (saved === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
  }
})();

document.addEventListener('DOMContentLoaded', function () {
  var btn = document.getElementById('theme-toggle');
  if (!btn) return;

  btn.addEventListener('click', function () {
    var isLight = document.documentElement.getAttribute('data-theme') === 'light';
    if (isLight) {
      document.documentElement.removeAttribute('data-theme');
      localStorage.setItem('ccm-theme', 'dark');
      btn.setAttribute('aria-label', 'Switch to light mode');
    } else {
      document.documentElement.setAttribute('data-theme', 'light');
      localStorage.setItem('ccm-theme', 'light');
      btn.setAttribute('aria-label', 'Switch to dark mode');
    }
  });

  // Set initial aria-label
  var isLight = document.documentElement.getAttribute('data-theme') === 'light';
  btn.setAttribute('aria-label', isLight ? 'Switch to dark mode' : 'Switch to light mode');
});
