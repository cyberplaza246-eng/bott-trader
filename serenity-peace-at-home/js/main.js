/**
 * Serenity Peace at Home — Main JavaScript
 */

(function () {
  'use strict';

  /* ========== Mobile Navigation ========== */
  const navToggle = document.querySelector('.nav-toggle');
  const mainNav = document.querySelector('.main-nav');

  if (navToggle && mainNav) {
    navToggle.addEventListener('click', function () {
      const expanded = navToggle.getAttribute('aria-expanded') === 'true';
      navToggle.setAttribute('aria-expanded', !expanded);
      mainNav.classList.toggle('open');
    });

    mainNav.querySelectorAll('a').forEach(function (link) {
      link.addEventListener('click', function () {
        navToggle.setAttribute('aria-expanded', 'false');
        mainNav.classList.remove('open');
      });
    });
  }

  /* ========== Header Scroll Effect ========== */
  const header = document.querySelector('.site-header');

  if (header) {
    window.addEventListener('scroll', function () {
      if (window.scrollY > 50) {
        header.classList.add('scrolled');
      } else {
        header.classList.remove('scrolled');
      }
    }, { passive: true });
  }

  /* ========== Active Nav Link ========== */
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.main-nav a').forEach(function (link) {
    const href = link.getAttribute('href');
    if (href === currentPage || (currentPage === '' && href === 'index.html')) {
      link.classList.add('active');
      link.setAttribute('aria-current', 'page');
    }
  });

  /* ========== Testimonial Carousel ========== */
  const carousel = document.querySelector('.testimonials');

  if (carousel) {
    const slides = carousel.querySelector('.testimonial-slides');
    const slideItems = carousel.querySelectorAll('.testimonial-slide');
    const prevBtn = carousel.querySelector('.testimonial-prev');
    const nextBtn = carousel.querySelector('.testimonial-next');
    const dotsContainer = carousel.querySelector('.testimonial-dots');
    let currentIndex = 0;
    let autoplayInterval;

    function goToSlide(index) {
      if (index < 0) index = slideItems.length - 1;
      if (index >= slideItems.length) index = 0;
      currentIndex = index;
      slides.style.transform = 'translateX(-' + (currentIndex * 100) + '%)';

      if (dotsContainer) {
        dotsContainer.querySelectorAll('.testimonial-dot').forEach(function (dot, i) {
          dot.classList.toggle('active', i === currentIndex);
          dot.setAttribute('aria-selected', i === currentIndex);
        });
      }
    }

    slideItems.forEach(function (_, i) {
      if (!dotsContainer) return;
      var dot = document.createElement('button');
      dot.className = 'testimonial-dot' + (i === 0 ? ' active' : '');
      dot.setAttribute('aria-label', 'Go to testimonial ' + (i + 1));
      dot.setAttribute('role', 'tab');
      dot.setAttribute('aria-selected', i === 0);
      dot.addEventListener('click', function () {
        goToSlide(i);
        resetAutoplay();
      });
      dotsContainer.appendChild(dot);
    });

    if (prevBtn) {
      prevBtn.addEventListener('click', function () {
        goToSlide(currentIndex - 1);
        resetAutoplay();
      });
    }

    if (nextBtn) {
      nextBtn.addEventListener('click', function () {
        goToSlide(currentIndex + 1);
        resetAutoplay();
      });
    }

    function startAutoplay() {
      autoplayInterval = setInterval(function () {
        goToSlide(currentIndex + 1);
      }, 6000);
    }

    function resetAutoplay() {
      clearInterval(autoplayInterval);
      startAutoplay();
    }

    startAutoplay();

    carousel.addEventListener('mouseenter', function () {
      clearInterval(autoplayInterval);
    });

    carousel.addEventListener('mouseleave', function () {
      startAutoplay();
    });
  }

  /* ========== Form Validation ========== */
  function validateForm(form) {
    var isValid = true;
    var fields = form.querySelectorAll('[data-required]');

    fields.forEach(function (field) {
      var errorEl = field.parentElement.querySelector('.error-message');
      var value = field.value.trim();

      field.classList.remove('error');

      if (errorEl) {
        errorEl.classList.remove('visible');
        errorEl.textContent = '';
      }

      if (!value) {
        isValid = false;
        field.classList.add('error');
        if (errorEl) {
          errorEl.textContent = 'This field is required.';
          errorEl.classList.add('visible');
        }
        return;
      }

      if (field.type === 'email' && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
        isValid = false;
        field.classList.add('error');
        if (errorEl) {
          errorEl.textContent = 'Please enter a valid email address.';
          errorEl.classList.add('visible');
        }
      }

      if (field.type === 'tel' && !/^[\d\s\-().+]{7,}$/.test(value)) {
        isValid = false;
        field.classList.add('error');
        if (errorEl) {
          errorEl.textContent = 'Please enter a valid phone number.';
          errorEl.classList.add('visible');
        }
      }
    });

    return isValid;
  }

  document.querySelectorAll('form[data-validate]').forEach(function (form) {
    form.addEventListener('submit', function (e) {
      e.preventDefault();

      var successEl = form.querySelector('.form-success');

      if (successEl) {
        successEl.classList.remove('visible');
      }

      if (!validateForm(form)) {
        var firstError = form.querySelector('.error');
        if (firstError) firstError.focus();
        return;
      }

      if (successEl) {
        successEl.classList.add('visible');
        successEl.setAttribute('role', 'alert');
      }

      form.reset();
      form.querySelectorAll('.error-message').forEach(function (el) {
        el.classList.remove('visible');
      });
      form.querySelectorAll('.error').forEach(function (el) {
        el.classList.remove('error');
      });

      if (successEl) {
        successEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    });

    form.querySelectorAll('[data-required]').forEach(function (field) {
      field.addEventListener('blur', function () {
        validateForm(form);
      });
    });
  });

  /* ========== Gallery Filters ========== */
  var filterBtns = document.querySelectorAll('.filter-btn');
  var galleryItems = document.querySelectorAll('.gallery-item');

  if (filterBtns.length && galleryItems.length) {
    filterBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        var filter = btn.getAttribute('data-filter');

        filterBtns.forEach(function (b) {
          b.classList.remove('active');
          b.setAttribute('aria-pressed', 'false');
        });
        btn.classList.add('active');
        btn.setAttribute('aria-pressed', 'true');

        galleryItems.forEach(function (item) {
          var category = item.getAttribute('data-category');
          if (filter === 'all' || category === filter) {
            item.style.display = '';
            item.style.opacity = '0';
            item.style.transform = 'scale(0.95)';
            requestAnimationFrame(function () {
              item.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
              item.style.opacity = '1';
              item.style.transform = 'scale(1)';
            });
          } else {
            item.style.display = 'none';
          }
        });
      });
    });
  }

  /* ========== Smooth Scroll for Anchor Links ========== */
  document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
    anchor.addEventListener('click', function (e) {
      var targetId = this.getAttribute('href');
      if (targetId === '#') return;

      var target = document.querySelector(targetId);
      if (target) {
        e.preventDefault();
        var headerOffset = header ? header.offsetHeight : 0;
        var top = target.getBoundingClientRect().top + window.scrollY - headerOffset;
        window.scrollTo({ top: top, behavior: 'smooth' });
      }
    });
  });
})();
