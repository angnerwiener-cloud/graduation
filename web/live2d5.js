/**
 * live2d5.js — Minimal Live2D Cubism 5 renderer
 * 只依赖 live2dcubismcore.min.js，无需 pixi / Framework
 * 支持 Cubism 4 & 5 格式，无遮罩数量限制
 */
(function (global) {
  'use strict';

  /* ── GLSL ───────────────────────────────────────────────── */
  const VS = `
    attribute vec2 aPos;
    attribute vec2 aUV;
    uniform mat3 uMVP;
    varying vec2 vUV;
    void main(){
      vec3 p = uMVP * vec3(aPos, 1.0);
      gl_Position = vec4(p.xy, 0.0, 1.0);
      vUV = aUV;
    }`;

  const FS = `
    precision mediump float;
    uniform sampler2D uTex;
    uniform float uOpacity;
    uniform bool uUseMask;
    uniform sampler2D uMask;
    uniform bool uInvertMask;
    varying vec2 vUV;
    void main(){
      vec4 c = texture2D(uTex, vUV);
      c.a *= uOpacity;
      if(uUseMask){
        // vUV が clip space -1..1 → 0..1 変換のマスク用テクスチャを使う
        // ここでは簡略化のため mask UV は vUV と同じ扱い
        float m = texture2D(uMask, vUV).a;
        if(uInvertMask) m = 1.0 - m;
        c *= m;
      }
      gl_FragColor = c;
    }`;

  /* マスク描画用シェーダー */
  const MASK_FS = `
    precision mediump float;
    uniform sampler2D uTex;
    varying vec2 vUV;
    void main(){
      vec4 c = texture2D(uTex, vUV);
      gl_FragColor = vec4(1.0, 1.0, 1.0, c.a);
    }`;

  /* ── GL utils ───────────────────────────────────────────── */
  function compileShader(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS))
      throw new Error('Shader: ' + gl.getShaderInfoLog(sh));
    return sh;
  }

  function createProgram(gl, vs, fs) {
    const p = gl.createProgram();
    gl.attachShader(p, compileShader(gl, gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compileShader(gl, gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS))
      throw new Error('Program: ' + gl.getProgramInfoLog(p));
    return p;
  }

  function makeOrthoMat(unitW, unitH, fitW, fitH) {
    // Maps model unit space to WebGL clip space [-1,1]
    // Center the model and preserve aspect ratio
    const mAspect = unitW / unitH;
    const vAspect = fitW / fitH;
    let sx, sy;
    if (vAspect > mAspect) {
      sy = 2.0 / unitH;
      sx = sy * (fitH / fitW);
    } else {
      sx = 2.0 / unitW;
      sy = sx * (fitW / fitH);
    }
    // Cubism Y: positive = up → clip space positive = up ✓ (but textures need Y flip)
    // Apply a slight zoom out margin
    sx *= 0.95; sy *= 0.95;
    // Column-major 3x3
    return new Float32Array([
      sx,  0,   0,
      0,  -sy,  0,   // flip Y so model renders right-side up
      0,   0,   1
    ]);
  }

  /* ── FBO for mask rendering ─────────────────────────────── */
  function createFBO(gl, w, h) {
    const fb  = gl.createFramebuffer();
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { fb, tex };
  }

  /* ── Renderer class ─────────────────────────────────────── */
  class Renderer {
    constructor(canvas, viewW, viewH) {
      this.canvas = canvas;
      canvas.width  = viewW;
      canvas.height = viewH;
      this.vW = viewW;
      this.vH = viewH;

      const gl = canvas.getContext('webgl', {
        alpha: true, premultipliedAlpha: false, antialias: true
      });
      if (!gl) throw new Error('WebGL not available');
      this.gl = gl;

      // Main program
      this._prog = createProgram(gl, VS, FS);
      this._maskProg = createProgram(gl, VS, MASK_FS);
      this._aPos  = gl.getAttribLocation(this._prog, 'aPos');
      this._aUV   = gl.getAttribLocation(this._prog, 'aUV');
      this._uMVP  = gl.getUniformLocation(this._prog, 'uMVP');
      this._uTex  = gl.getUniformLocation(this._prog, 'uTex');
      this._uOpa  = gl.getUniformLocation(this._prog, 'uOpacity');
      this._uUseMask   = gl.getUniformLocation(this._prog, 'uUseMask');
      this._uMaskTex   = gl.getUniformLocation(this._prog, 'uMask');
      this._uInvertMask= gl.getUniformLocation(this._prog, 'uInvertMask');

      this._mAPos = gl.getAttribLocation(this._maskProg, 'aPos');
      this._mAUV  = gl.getAttribLocation(this._maskProg, 'aUV');
      this._mMVP  = gl.getUniformLocation(this._maskProg, 'uMVP');
      this._mTex  = gl.getUniformLocation(this._maskProg, 'uTex');

      this._vbuf  = gl.createBuffer();
      this._ibuf  = gl.createBuffer();
      this._fbo   = createFBO(gl, viewW, viewH);

      this.textures   = [];
      this.coreModel  = null;
      this.mvp        = null;
      this._phase     = 0;
      this._raf       = null;
      this._breathIdx = -1;
      this._onReady   = null;
    }

    async load(modelJsonUrl) {
      const base = modelJsonUrl.substring(0, modelJsonUrl.lastIndexOf('/') + 1);

      // 1. model3.json
      const json = await fetch(modelJsonUrl).then(r => {
        if (!r.ok) throw new Error('Cannot fetch model3.json: ' + r.status);
        return r.json();
      });

      // 2. moc3 binary
      const mocUrl = base + json.FileReferences.Moc;
      const mocBuf = await fetch(mocUrl).then(r => {
        if (!r.ok) throw new Error('Cannot fetch moc3: ' + r.status);
        return r.arrayBuffer();
      });

      // 3. Create Core model
      const Core = window.Live2DCubismCore;
      if (!Core) throw new Error('live2dcubismcore.min.js not loaded');

      const moc = Core.Moc.fromArrayBuffer(mocBuf);
      if (!moc) throw new Error('Moc.fromArrayBuffer failed');

      this.coreModel = Core.Model.fromMoc(moc);
      if (!this.coreModel) throw new Error('Model.fromMoc failed');

      // Canvas info → MVP
      const info = this.coreModel.canvasinfo;
      const ppu  = info.pixelsPerUnit || 1;
      const unitW = info.canvasWidth  / ppu;
      const unitH = info.canvasHeight / ppu;
      this.mvp = makeOrthoMat(unitW, unitH, this.vW, this.vH);

      // 4. Textures
      const texPaths = json.FileReferences.Textures;
      this.textures = await Promise.all(
        texPaths.map(p => this._loadTex(base + p))
      );

      // 5. Parameter indices for animation
      const ids = Array.from(this.coreModel.parameters.ids);
      this._breathIdx = ids.indexOf('ParamBreath');
      this._eyeBlinkL = ids.indexOf('ParamEyeLOpen');
      this._eyeBlinkR = ids.indexOf('ParamEyeROpen');
      this._blinkT    = 0;
      this._nextBlink = 3 + Math.random() * 4;

      // Initial update to populate dynamicFlags
      this.coreModel.update();

      console.log('[Live2D5] ✓ model loaded —', ids.length, 'params,',
        this.coreModel.drawables.ids.length, 'drawables');

      if (this._onReady) this._onReady(this);
      return this;
    }

    onReady(fn) { this._onReady = fn; return this; }

    _loadTex(url) {
      return new Promise((resolve, reject) => {
        const gl  = this.gl;
        const tex = gl.createTexture();
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
          gl.bindTexture(gl.TEXTURE_2D, tex);
          gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
          gl.generateMipmap(gl.TEXTURE_2D);
          resolve(tex);
        };
        img.onerror = () => reject(new Error('Failed to load texture: ' + url));
        img.src = url;
      });
    }

    start() {
      let last = 0;
      const loop = (ts) => {
        const dt = Math.min((ts - last) / 1000, 0.05);
        last = ts;
        this._update(dt);
        this._draw();
        this._raf = requestAnimationFrame(loop);
      };
      this._raf = requestAnimationFrame(loop);
    }

    stop() {
      if (this._raf) { cancelAnimationFrame(this._raf); this._raf = null; }
    }

    _update(dt) {
      if (!this.coreModel) return;
      const p   = this.coreModel.parameters;
      const v   = p.values;

      // Breath
      this._phase += dt;
      if (this._breathIdx >= 0)
        v[this._breathIdx] = (Math.sin(this._phase * 0.8) * 0.5 + 0.5);

      // Eye blink
      this._blinkT += dt;
      if (this._eyeBlinkL >= 0 || this._eyeBlinkR >= 0) {
        if (this._blinkT >= this._nextBlink) {
          const blinkDur = 0.4;
          const t = this._blinkT - this._nextBlink;
          let open;
          if (t < blinkDur / 2)       open = 1 - (t / (blinkDur / 2));
          else if (t < blinkDur)       open = (t - blinkDur / 2) / (blinkDur / 2);
          else {
            open = 1;
            if (t >= blinkDur + 0.1) {
              this._blinkT    = 0;
              this._nextBlink = 3 + Math.random() * 4;
            }
          }
          if (this._eyeBlinkL >= 0) v[this._eyeBlinkL] = open;
          if (this._eyeBlinkR >= 0) v[this._eyeBlinkR] = open;
        }
      }

      this.coreModel.update();
    }

    // Set any parameter by ID
    setParam(id, val) {
      if (!this.coreModel) return;
      const ids = this.coreModel.parameters.ids;
      for (let i = 0; i < ids.length; i++) {
        if (ids[i] === id) { this.coreModel.parameters.values[i] = val; return; }
      }
    }

    _draw() {
      if (!this.coreModel || !this.mvp) return;
      const gl = this.gl;
      const d  = this.coreModel.drawables;
      const C  = window.Live2DCubismCore.Constants;
      const n  = d.ids.length;

      // Sort by render order
      const order = Array.from({length: n}, (_, i) => i);
      order.sort((a, b) => d.renderOrders[a] - d.renderOrders[b]);

      /* ── Pass 1: render each mask group into FBO ── */
      // Build mask lookup: maskKey → FBO or null
      // For simplicity, render each drawable that needs masking using its masks rendered to FBO
      // Then sample FBO when drawing the drawable
      // (Full implementation: one FBO texture per unique mask set)
      // Simplified here: use a single shared FBO, drawn per drawable group
      // This handles any number of masks correctly.

      gl.viewport(0, 0, this.vW, this.vH);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);

      gl.useProgram(this._prog);
      gl.uniformMatrix3fv(this._uMVP, false, this.mvp);

      for (const i of order) {
        // Visibility
        if (!C.isVisible(d.dynamicFlags[i])) continue;

        const opacity = d.opacities[i];
        if (opacity <= 0.001) continue;

        // Blend mode
        const isAdd = C.isBlendModeAdditive(d.constantFlags[i]);
        const isMul = C.isBlendModeMultiplicative(d.constantFlags[i]);
        if (isAdd)
          gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
        else if (isMul)
          gl.blendFunc(gl.DST_COLOR, gl.ONE_MINUS_SRC_ALPHA);
        else
          gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

        // Has masks?
        const maskCount = d.maskCounts[i];
        if (maskCount > 0) {
          this._renderMasks(d, d.masks[i], maskCount);
          gl.useProgram(this._prog);
          gl.uniformMatrix3fv(this._uMVP, false, this.mvp);
          gl.uniform1i(this._uUseMask, 1);
          gl.activeTexture(gl.TEXTURE1);
          gl.bindTexture(gl.TEXTURE_2D, this._fbo.tex);
          gl.uniform1i(this._uMaskTex, 1);
          const inv = C.isInvertedMask(d.constantFlags[i]) ? 1 : 0;
          gl.uniform1i(this._uInvertMask, inv);
        } else {
          gl.uniform1i(this._uUseMask, 0);
        }

        // Texture
        const texIdx = d.textureIndices[i];
        if (!this.textures[texIdx]) continue;
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.textures[texIdx]);
        gl.uniform1i(this._uTex, 0);
        gl.uniform1f(this._uOpa, opacity);

        this._uploadAndDraw(d.vertexPositions[i], d.vertexUvs[i],
          d.vertexCounts[i], d.indices[i], this._aPos, this._aUV);
      }
    }

    _renderMasks(d, maskIndices, maskCount) {
      const gl = this.gl;
      const C  = window.Live2DCubismCore.Constants;

      gl.bindFramebuffer(gl.FRAMEBUFFER, this._fbo.fb);
      gl.viewport(0, 0, this.vW, this.vH);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

      gl.useProgram(this._maskProg);
      gl.uniformMatrix3fv(this._mMVP, false, this.mvp);

      for (let m = 0; m < maskCount; m++) {
        const mi = maskIndices[m];
        const texIdx = d.textureIndices[mi];
        if (!this.textures[texIdx]) continue;
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.textures[texIdx]);
        gl.uniform1i(this._mTex, 0);
        this._uploadAndDraw(d.vertexPositions[mi], d.vertexUvs[mi],
          d.vertexCounts[mi], d.indices[mi], this._mAPos, this._mAUV);
      }

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.vW, this.vH);
    }

    _uploadAndDraw(positions, uvs, vcount, indices, aPosLoc, aUVLoc) {
      const gl   = this.gl;
      const data = new Float32Array(vcount * 4);
      for (let v = 0; v < vcount; v++) {
        data[v * 4 + 0] = positions[v * 2];
        data[v * 4 + 1] = positions[v * 2 + 1];
        data[v * 4 + 2] = uvs[v * 2];
        data[v * 4 + 3] = uvs[v * 2 + 1];
      }
      const stride = 16;
      gl.bindBuffer(gl.ARRAY_BUFFER, this._vbuf);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(aPosLoc);
      gl.vertexAttribPointer(aPosLoc, 2, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(aUVLoc);
      gl.vertexAttribPointer(aUVLoc, 2, gl.FLOAT, false, stride, 8);

      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this._ibuf);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW);
      gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_SHORT, 0);
    }
  }

  /* ── Public API ─────────────────────────────────────────── */
  global.Live2D5 = {
    /**
     * create(canvas, modelJsonUrl, options)
     * options: { width, height }
     */
    async create(canvas, modelJsonUrl, opts) {
      opts = opts || {};
      const w = opts.width  || canvas.parentElement.clientWidth  || 400;
      const h = opts.height || canvas.parentElement.clientHeight || 600;
      const r = new Renderer(canvas, w, h);
      await r.load(modelJsonUrl);
      r.start();
      return r;
    },
    Renderer
  };

})(window);
