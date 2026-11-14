// 每页左侧的目录：从各节的 h2 自动生成，滚动时高亮当前一节。宽屏固定在左边，窄屏收成页面顶部的“目录”。
(function () {
  const heads = [...document.querySelectorAll('.wrap section h2')];
  if (heads.length < 2) return;
  const toc = document.createElement('nav');
  toc.className = 'toc';
  toc.setAttribute('aria-label', '本页目录');
  const det = document.createElement('details');
  const sum = document.createElement('summary');
  sum.textContent = '本页目录';
  const ol = document.createElement('ol');
  const links = heads.map((h, i) => {
    if (!h.id) h.id = 's' + (i + 1);
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = h.textContent.replace(/\s+/g, ' ').trim();
    li.appendChild(a);
    ol.appendChild(li);
    return a;
  });
  det.append(sum, ol);
  toc.appendChild(det);
  const wrap = document.querySelector('.wrap');
  wrap.parentNode.insertBefore(toc, wrap);
  document.body.classList.add('has-toc');

  const wide = window.matchMedia('(min-width: 1100px)');
  const sync = () => { if (wide.matches) det.open = true; };
  sync();
  wide.addEventListener ? wide.addEventListener('change', sync) : wide.addListener(sync);
  ol.addEventListener('click', (e) => { if (e.target.closest('a') && !wide.matches) det.open = false; });

  // 当前一节：最后一个顶部已经滚过视口上方 1/3 的标题
  const mark = () => {
    const y = window.innerHeight / 3;
    let k = 0;
    heads.forEach((h, i) => { if (h.getBoundingClientRect().top < y) k = i; });
    links.forEach((a, i) => a.classList.toggle('on', i === k));
  };
  window.addEventListener('scroll', mark, { passive: true });
  mark();
})();
