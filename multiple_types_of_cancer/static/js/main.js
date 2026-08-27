// Multiple Types of Cancer - Mini Project
// Shared front-end behaviour: nav toggle, scroll-reveal, animated bars.

document.addEventListener('DOMContentLoaded', function () {
  // Mobile nav toggle
  var toggle = document.querySelector('.nav-toggle');
  var links = document.querySelector('.nav-links');
  if (toggle && links) {
    toggle.addEventListener('click', function () {
      links.classList.toggle('open');
    });
    links.querySelectorAll('a').forEach(function (a) {
      a.addEventListener('click', function () { links.classList.remove('open'); });
    });
  }

  // Reveal-on-scroll
  var revealEls = document.querySelectorAll('.animate-in');
  if ('IntersectionObserver' in window && revealEls.length) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('in-view');
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });
    revealEls.forEach(function (el) { io.observe(el); });
  } else {
    revealEls.forEach(function (el) { el.classList.add('in-view'); });
  }

  // Animate any width-based bars (confidence / class distribution) once visible
  var bars = document.querySelectorAll('[data-fill]');
  if (bars.length) {
    var barIo = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          var target = entry.target;
          target.style.width = target.getAttribute('data-fill') + '%';
          barIo.unobserve(target);
        }
      });
    }, { threshold: 0.2 });
    bars.forEach(function (bar) { barIo.observe(bar); });
  }
});

// Drag & drop + live preview for the detect page
function initUploader() {
  var dropzone = document.getElementById('dropzone');
  var fileInput = document.getElementById('imagefile');
  var preview = document.getElementById('preview-image');
  var wrap = document.getElementById('preview-wrap');
  var chip = document.getElementById('file-name');
  var submitBtn = document.getElementById('submit-btn');

  if (!dropzone || !fileInput) return;

  function showFile(file) {
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function () {
      preview.src = reader.result;
      wrap.hidden = false;
      chip.hidden = false;
      chip.textContent = file.name;
      submitBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  dropzone.addEventListener('click', function () { fileInput.click(); });

  fileInput.addEventListener('change', function () {
    showFile(fileInput.files[0]);
  });

  ['dragenter', 'dragover'].forEach(function (evt) {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(function (evt) {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', function (e) {
    var file = e.dataTransfer.files[0];
    if (file) {
      fileInput.files = e.dataTransfer.files;
      showFile(file);
    }
  });
}

// Tab switcher for the "browse the dataset" sample picker on the Detect page
function initPicker() {
  var tabs = document.querySelectorAll('.picker-tab');
  var panels = document.querySelectorAll('.picker-panel');
  if (!tabs.length) return;

  tabs.forEach(function (tab) {
    tab.addEventListener('click', function () {
      tabs.forEach(function (t) { t.classList.remove('active'); });
      panels.forEach(function (p) { p.classList.remove('active'); });
      tab.classList.add('active');
      var target = document.getElementById(tab.getAttribute('data-target'));
      if (target) target.classList.add('active');
    });
  });
}
