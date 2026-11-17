# pairwise-lightmap-bake

烘焙一张大场景 lightmap 时，UV 重叠导致同一块光照被烤到两个面；texel density 没有按物体尺寸分配，小物件占了大片图集；lightmap seam 在 UV 边界漏光，dilation 只扩了一圈；AO 采样用了固定半球方向，室内角落出现噪点；烘焙时间随 light 数量线性增长，没有做 cluster 剔除；光源移动后没有增量更新，整张图重烤。gi/lightmap.py 是核心，tests/test_lightmap.py 15个红。要修的是：检测 UV 重叠并重新打包；按世界尺寸算 texel density；seam 做多圈 dilation；AO 用 cosine-weighted 采样并降噪；按 cluster 剔除不可见光源；支持增量重烤。

## 运行

```bash
python -m pytest tests -q
```

初始状态：15 个用例失败。修复实现使全部用例通过，不要修改 `tests/` 下的测试。

语言/框架：Python 3, pytest；环境可复现等级：无外部依赖。
