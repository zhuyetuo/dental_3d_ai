// 图的三种看法：静态图 / 动图 / 3D 交互。.views[data-src] 指向 assets 里导出的数据（make_explain_bite.py）。
(function () {
  const C = { crown: '#33c3d1', crownEdge: '#0f7d89', ant: '#aab8d3', antEdge: '#6c7c9f', bad: '#ff5a5a', good: '#2fd08a', warn: '#f5b74f', txt: '#e8eef8', bg: '#0b1426' };
  const cache = {};
  function data(src) {
    if (!cache[src]) {
      cache[src] = Promise.all([fetch(src + '.json').then(r => r.json()), fetch(src + '.bin').then(r => r.arrayBuffer())]);
    }
    return cache[src];
  }

  document.querySelectorAll('.views').forEach(box => {
    const btns = box.querySelectorAll('.vbtn');
    const panes = box.querySelectorAll('.vpane');
    const started = {};
    btns.forEach(b => b.addEventListener('click', () => {
      const v = b.dataset.view;
      btns.forEach(x => x.classList.toggle('on', x === b));
      panes.forEach(p => { p.hidden = p.dataset.view !== v; });
      box.dataset.active = v;
      if (!started[v]) {
        started[v] = true;
        if (v === 'anim') startAnim(box);
        if (v === '3d') start3d(box);
      }
    }));
  });

  // ---------- 动图：剖面上牙冠上下移动 ----------
  function startAnim(box) {
    const cv = box.querySelector('.vpane[data-view="anim"] canvas');
    const info = box.querySelector('.vpane[data-view="anim"] .vstate');
    data(box.dataset.src).then(([m]) => {
      const g = m.grid, cr = m.crown, an = m.ant, gap0 = m.gap0;
      // 画面中心对准最窄处（接触的地方），横竖等比例：上下各留约 3 mm
      let iMin = 0, best = Infinity;
      for (let i = 0; i < g.length; i++) if (cr[i] !== null && an[i] !== null && an[i] - cr[i] < best) { best = an[i] - cr[i]; iMin = i; }
      const vc = an[iMin], span = (g[g.length - 1] - g[0]) * 0.46;
      const lo = vc - span * 0.58, hi = vc + span * 0.42;
      let paused = false, t0 = performance.now(), tp = 0;
      cv.addEventListener('click', () => { paused = !paused; if (!paused) t0 = performance.now() - tp; });
      function frame(now) {
        if (box.dataset.active !== 'anim') { requestAnimationFrame(frame); return; }
        if (!paused) tp = now - t0;
        // 一个周期 6 秒：离开 0.8 mm → 刚好接触 → 穿进 0.5 mm → 回去
        const ph = (tp / 6000) % 1, w = 0.5 - 0.5 * Math.cos(2 * Math.PI * ph);
        const s = gap0 - 0.8 + w * 1.3;
        draw(s);
        requestAnimationFrame(frame);
      }
      function draw(s) {
        const W = cv.clientWidth, H = Math.round(W * 0.46);
        if (cv.width !== W * 2) { cv.width = W * 2; cv.height = H * 2; cv.style.height = H + 'px'; }
        const ctx = cv.getContext('2d');
        ctx.setTransform(2, 0, 0, 2, 0, 0);
        ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
        const X = u => (u - g[0]) / (g[g.length - 1] - g[0]) * W;
        const Y = v => H - (v - lo) / (hi - lo) * H;
        const path = (arr, off, to) => {
          ctx.beginPath(); let started = false, first = null, last = null;
          arr.forEach((v, i) => { if (v === null) return; const x = X(g[i]), y = Y(v + off); if (!started) { ctx.moveTo(x, Y(to)); ctx.lineTo(x, y); started = true; first = x; } else ctx.lineTo(x, y); last = x; });
          ctx.lineTo(last, Y(to)); ctx.closePath();
        };
        path(an, 0, hi + 5); ctx.fillStyle = C.ant; ctx.fill(); ctx.strokeStyle = C.antEdge; ctx.lineWidth = 2; ctx.stroke();
        path(cr, s, lo - 5); ctx.fillStyle = C.crown; ctx.fill(); ctx.strokeStyle = C.crownEdge; ctx.stroke();
        let minGap = Infinity;
        for (let i = 0; i < g.length - 1; i++) {
          if (cr[i] === null || an[i] === null) continue;
          const d = an[i] - (cr[i] + s);
          minGap = Math.min(minGap, d);
          if (d < 0) { ctx.fillStyle = C.bad; ctx.fillRect(X(g[i]), Y(cr[i] + s), X(g[i + 1]) - X(g[i]) + 0.5, Y(an[i]) - Y(cr[i] + s)); }
          else if (d <= 0.06 && i % 5 === 0) { ctx.fillStyle = C.good; ctx.beginPath(); ctx.arc(X(g[i]), Y(cr[i] + s), 5, 0, 7); ctx.fill(); }
        }
        let txt, col;
        if (minGap > 0.06) { txt = '没咬上：空 ' + minGap.toFixed(2) + ' mm'; col = C.warn; }
        else if (minGap >= -0.06) { txt = '刚好接触 ✓'; col = C.good; }
        else { txt = '穿进对颌 ' + (-minGap).toFixed(2) + ' mm'; col = C.bad; }
        ctx.font = '700 22px Inter, "PingFang SC", "Microsoft YaHei", sans-serif';
        const tw = ctx.measureText(txt).width;
        ctx.fillStyle = 'rgba(11,20,38,.88)'; ctx.fillRect(10, 10, tw + 24, 38);
        ctx.fillStyle = col; ctx.fillText(txt, 22, 37);
        ctx.font = '700 15px Inter, "PingFang SC", "Microsoft YaHei", sans-serif';
        ctx.fillStyle = '#1d2a44'; ctx.fillText('对颌（上面的牙）', W - 150, 32);
        ctx.fillStyle = '#06343a'; ctx.fillText('牙冠', W / 2 - 15, H - 18);
        ctx.font = '600 12px Inter, "PingFang SC", "Microsoft YaHei", sans-serif';
        ctx.fillStyle = 'rgba(11,20,38,.75)'; ctx.fillRect(W - 160, H - 30, 150, 22);
        ctx.fillStyle = '#e8eef8'; ctx.fillText('点一下画面暂停 / 继续', W - 152, H - 14);
        if (info) { info.textContent = txt; info.style.color = col; }
      }
      requestAnimationFrame(frame);
    });
  }

  // ---------- 3D 交互：真实牙冠 + 对颌，滑块调高低，按到对颌的距离上色 ----------
  function loadScript(src) {
    return new Promise((ok, bad) => { const s = document.createElement('script'); s.src = src; s.onload = ok; s.onerror = bad; document.head.appendChild(s); });
  }
  function start3d(box) {
    const host = box.querySelector('.v3d');
    const slider = box.querySelector('.vpane[data-view="3d"] input[type=range]');
    const out = box.querySelector('.vpane[data-view="3d"] output');
    host.textContent = '加载中…';
    const libs = window.THREE ? Promise.resolve() : loadScript('https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js')
      .then(() => loadScript('https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js'));
    Promise.all([libs, data(box.dataset.src)]).then(([, [m, buf]]) => {
      host.textContent = '';
      let o = 0;
      const take = (T, n) => { const a = new T(buf, o, n); o += n * T.BYTES_PER_ELEMENT; return a; };
      const cv = take(Float32Array, m.crown_verts * 3), cf = take(Uint32Array, m.crown_faces * 3), cd = take(Float32Array, m.crown_verts);
      const av = take(Float32Array, m.ant_verts * 3), af = take(Uint32Array, m.ant_faces * 3);
      const W = host.clientWidth, H = Math.round(W * 0.55);
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(Math.min(2, window.devicePixelRatio)); renderer.setSize(W, H); renderer.setClearColor(0x0b1426);
      host.appendChild(renderer.domElement);
      const scene = new THREE.Scene();
      const cam = new THREE.PerspectiveCamera(32, W / H, 0.1, 400);
      let top = -Infinity;  // 牙冠最高处：相机对准咬合接触的高度
      for (let i = 1; i < m.crown_verts * 3; i += 3) top = Math.max(top, cv[i]);
      cam.position.set(22, top + 14, 38);
      const ctl = new THREE.OrbitControls(cam, renderer.domElement); ctl.target.set(0, top - 1, 0); ctl.enableDamping = true; ctl.update();
      scene.add(new THREE.HemisphereLight(0xffffff, 0x334466, 0.9));
      const dl = new THREE.DirectionalLight(0xffffff, 0.7); dl.position.set(10, 20, 15); scene.add(dl);
      function mesh(v, f, mat) {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(v, 3)); geo.setIndex(new THREE.BufferAttribute(f, 1)); geo.computeVertexNormals();
        const me = new THREE.Mesh(geo, mat); scene.add(me); return me;
      }
      const antMesh = mesh(av, af, new THREE.MeshStandardMaterial({ color: 0xaab8d3, roughness: 0.75, side: THREE.DoubleSide, transparent: true, opacity: 0.55 }));
      const colors = new Float32Array(m.crown_verts * 3);
      const crown = mesh(cv, cf, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6, side: THREE.DoubleSide }));
      crown.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      const minD = Math.min(...cd);
      const col = (d, i) => {  // 蓝：有间隙；绿：刚好接触；红：穿进去
        let r, g, b;
        if (d < -0.05) { r = 1.0; g = 0.3; b = 0.3; }                       // 红：穿进去
        else if (d <= 0.1) { r = 0.18; g = 0.85; b = 0.5; }                 // 绿：刚好接触
        else if (d <= 0.8) { const t = (d - 0.1) / 0.7; r = 0.25 - 0.05 * t; g = 0.4 + 0.37 * t; b = 0.98 - 0.16 * t; }  // 蓝 → 青：有间隙
        else { r = 0.2; g = 0.77; b = 0.82; }                                // 青：离对颌远（牙冠本色）
        colors[3 * i] = r; colors[3 * i + 1] = g; colors[3 * i + 2] = b;
      };
      function setGap(gap) {  // gap：牙冠离对颌最近处的目标距离（负数 = 穿进去）
        const s = minD - gap;
        crown.position.y = s;
        for (let i = 0; i < m.crown_verts; i++) col(cd[i] - s, i);
        crown.geometry.attributes.color.needsUpdate = true;
        out.textContent = gap > 0.06 ? `没咬上：空 ${gap.toFixed(2)} mm` : gap >= -0.06 ? '刚好接触 ✓' : `穿进对颌 ${(-gap).toFixed(2)} mm`;
        out.style.color = gap > 0.06 ? C.warn : gap >= -0.06 ? C.good : C.bad;
      }
      slider.addEventListener('input', () => setGap(parseFloat(slider.value)));
      setGap(parseFloat(slider.value));
      const opa = box.querySelector('.vpane[data-view="3d"] .vopa');
      if (opa) opa.addEventListener('click', () => { antMesh.material.opacity = antMesh.material.opacity < 0.9 ? 1 : 0.55; antMesh.material.transparent = antMesh.material.opacity < 1; });
      (function loop() { if (box.dataset.active === '3d') { ctl.update(); renderer.render(scene, cam); } requestAnimationFrame(loop); })();
      window.addEventListener('resize', () => { const w = host.clientWidth, h = Math.round(w * 0.55); renderer.setSize(w, h); cam.aspect = w / h; cam.updateProjectionMatrix(); });
    }).catch(e => { host.textContent = '3D 加载失败：' + e; });
  }
})();
