# Waterful Ring-Toss 3D (1976 经典掌上水压套圈机)

基于 **Three.js** 完整复刻 1976 年 **TOMY** 纯物理无电池水压套圈掌机（Waterful Ring Toss）。

## 特性

- **真实流体动力学** — 手写 WASM 线性内存物理加速内核，模拟水泵喷流、水波（SWE）、惯性与阻尼
- **塑料环各向异性水阻** — 环面（Torus）在不同姿态下受水阻不同，翻面时升力/阻力随角度变化
- **三维圆环与立柱碰撞** — 圆环可真正套上立柱（Gauss 环绕数判定）、可斜着卡住、可倾倒滑脱
- **IPC C² 对数势垒** — 环与环、环与立柱的增量势能接触求解，防穿模、防抖
- **手机陀螺仪重力感应** — 摇晃机身带动整罐水晃动，倾斜改变重力方向（`sensorBridge` 桥接 DeviceOrientation）
- **无电池、无网络** — 纯物理模拟，所有计算在浏览器本地完成

## 在线游玩

Live: **https://hana-ame.github.io/waterful-ring-toss-3d/**

## 本地运行

**前置要求：** Node.js 20+

```bash
npm install
npm run dev        # http://localhost:3000
```

## 构建与测试

```bash
npm run lint       # TypeScript 类型检查
npm test           # 物理引擎测试套件（11 项）
npm run build      # 生产构建到 dist/
```

> 测试通过 `npx --yes tsx --test src/physics/waterPhysics.test.ts` 运行（见 `.github/workflows/deploy-gh-pages.yml`）。

## 部署

仓库通过 **GitHub Actions** 在每次 push 到 `main` 时自动构建并部署到 GitHub Pages：

1. 安装依赖（`npm ci`）
2. 运行物理单元测试
3. `vite build` 构建到 `dist/`
4. `ghPagesHardArmorPlugin` 注入 `.nojekyll` 与 SPA 回退 `404.html`
5. 上传 `dist/` 作为 Pages 产物并部署

纯静态单页应用，无后端、无 API Key。物理引擎（含 WASM 内核）全部在浏览器内运行。
