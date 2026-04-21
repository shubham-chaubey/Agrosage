// AgroSage 2.0 — main.js
'use strict';

document.addEventListener('DOMContentLoaded', () => {

  const html = document.documentElement;

  // ══════════════════════════════════════════════
  // 1. DARK MODE
  // ══════════════════════════════════════════════
  const themeBtn  = document.getElementById('themeBtn');
  const iconSun   = themeBtn?.querySelector('.icon-sun');
  const iconMoon  = themeBtn?.querySelector('.icon-moon');

  // Load saved theme (default: light)
  const savedTheme = localStorage.getItem('as_theme') || 'light';
  applyTheme(savedTheme);

  function applyTheme(theme) {
    html.setAttribute('data-theme', theme);
    localStorage.setItem('as_theme', theme);
    if (iconSun && iconMoon) {
      iconSun.style.display  = theme === 'dark' ? 'none'  : 'block';
      iconMoon.style.display = theme === 'dark' ? 'block' : 'none';
    }
  }

  themeBtn?.addEventListener('click', () => {
    const current = html.getAttribute('data-theme');
    applyTheme(current === 'dark' ? 'light' : 'dark');
  });


  // ══════════════════════════════════════════════
  // 2. LANGUAGE SWITCHER (EN / HI)
  // ══════════════════════════════════════════════
  const langBtns   = document.querySelectorAll('#langSwitch .lang-btn');
  const savedLang  = localStorage.getItem('as_lang') || 'en';
  applyLang(savedLang);

  function applyLang(lang) {
    html.setAttribute('data-lang', lang);
    localStorage.setItem('as_lang', lang);

    // Activate correct button
    langBtns.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.val === lang);
    });

    // Translate every element that has data-en / data-hi
    document.querySelectorAll('[data-en]').forEach(el => {
      const text = el.getAttribute(`data-${lang}`);
      if (text) {
        // Don't overwrite inner HTML that has child elements
        if (el.children.length === 0) el.textContent = text;
      }
    });

    // Swap placeholder on chat textarea if present
    const chatTA = document.getElementById('questionInput');
    if (chatTA) {
      chatTA.placeholder = lang === 'hi'
        ? 'अपना कृषि सवाल यहाँ टाइप करें...'
        : 'Type your farming question in Hindi or English...';
    }

    // Swap html lang attr for accessibility
    html.setAttribute('lang', lang === 'hi' ? 'hi' : 'en');
  }

  langBtns.forEach(btn => {
    btn.addEventListener('click', () => applyLang(btn.dataset.val));
  });


  // ══════════════════════════════════════════════
  // 3. MOBILE MENU
  // ══════════════════════════════════════════════
  const mobileBtn  = document.getElementById('mobileMenuBtn');
  const mobileMenu = document.getElementById('mobileMenu');

  mobileBtn?.addEventListener('click', () => mobileMenu?.classList.toggle('open'));
  document.addEventListener('click', e => {
    if (mobileBtn && mobileMenu &&
        !mobileBtn.contains(e.target) &&
        !mobileMenu.contains(e.target)) {
      mobileMenu.classList.remove('open');
    }
  });


  // ══════════════════════════════════════════════
  // 4. AUTO-DISMISS FLASH MESSAGES
  // ══════════════════════════════════════════════
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity .5s';
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 500);
    }, 4500);
  });


  // ══════════════════════════════════════════════
  // 5. CONFIDENCE BAR ANIMATION (Recommendation page)
  // ══════════════════════════════════════════════
  const fill = document.querySelector('.conf-bar-fill');
  if (fill) {
    const target = fill.style.width;
    fill.style.width = '0%';
    requestAnimationFrame(() =>
      setTimeout(() => { fill.style.width = target; }, 150)
    );
  }


  // ══════════════════════════════════════════════
  // 6. PASSWORD TOGGLE (Login / Signup pages)
  // ══════════════════════════════════════════════
  document.querySelectorAll('.toggle-pwd').forEach(btn => {
    btn.addEventListener('click', () => {
      const input = document.getElementById(btn.dataset.target);
      if (!input) return;
      input.type = input.type === 'password' ? 'text' : 'password';
      btn.textContent = input.type === 'password' ? '👁' : '🙈';
    });
  });


  // ══════════════════════════════════════════════
  // 7. RECOMMENDATION FORM — spinner + sample fill
  // ══════════════════════════════════════════════
  const recForm   = document.getElementById('recForm');
  const submitBtn = document.getElementById('submitBtn');
  const spinner   = document.getElementById('btnSpinner');

  recForm?.addEventListener('submit', () => {
    if (spinner)   spinner.style.display = 'inline-block';
    if (submitBtn) submitBtn.disabled = true;
  });

  document.querySelectorAll('.sample-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const fields = { nitrogen: btn.dataset.n, phosphorus: btn.dataset.p,
        potassium: btn.dataset.k, temperature: btn.dataset.t,
        humidity: btn.dataset.h, ph: btn.dataset.ph, rainfall: btn.dataset.r };
      Object.entries(fields).forEach(([id, val]) => {
        const el = document.getElementById(id);
        if (el) el.value = val;
      });
    });
  });

});
