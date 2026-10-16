---
layout: compress
# WARNING: Don't use '//' to comment out code, use '{% comment %}' and '{% endcomment %}' instead.
# Site-level override of the theme's assets/js/data/mathjax.js: backport of upstream commit 99614ae
# (fix(math): restore formula overflow scroll for MathJax 4), which jekyll-theme-chirpy 7.6.0 lacks.
# Delete this file once the theme version you use includes that fix.
---

{%- comment -%}
  See: <https://docs.mathjax.org/en/latest/options/input/tex.html#tex-options>
{%- endcomment -%}

MathJax = {
  tex: {
    {%- comment -%} start/end delimiter pairs for in-line math {%- endcomment -%}
    inlineMath: [
      ['$', '$'],
      ['\\(', '\\)']
    ],
    {%- comment -%} start/end delimiter pairs for display math {%- endcomment -%}
    displayMath: [
      ['$$', '$$'],
      ['\\[', '\\]']
    ],
    {%- comment -%} equation numbering {%- endcomment -%}
    tags: 'ams'
  },
  output: {
    {%- comment -%} scroll long display equations instead of overflowing the page {%- endcomment -%}
    displayOverflow: 'scroll'
  }
};
