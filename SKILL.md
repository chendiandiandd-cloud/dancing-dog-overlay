---
name: dancing-dog-overlay
description: 为用户上传的多张图片、多段视频或混合素材叠加预制跳舞卡通小狗，不调用图像或视频生成模型。支持离线透明动画缓存、尺寸与画面候选落点联合自适应、分镜检测与切镜换狗、种子复现、手动落点、保留原视频音轨和逐文件 MP4 输出。适用于“加一只跳舞小狗”“照片变小狗跳舞视频”“每个镜头换狗”“多素材批量套跳舞小狗特效”等需求。
author: 'chenyanqi.1008'
---

# 跳舞小狗叠加

## 效果约定

- 直接使用随包的 16 只动画小狗，合成成片，不调用生成模型，也不在运行时调用视觉大模型。
- 多文件默认独立输出；不把用户上传的素材擅自拼成一条视频。
- 每张图片默认独立生成 5 秒 MP4，视频保持原时长并保留原音轨，小狗素材静音。
- 每个分镜只新增一只小狗；相邻分镜使用不同狗，镜头内部不换狗。
- 不识别或清除输入素材中已经存在的小狗，参考效果视频不可当作没有贴纸的原片。
- 默认联合评估尺寸与落点，常规画布高度约画高0.18–0.24（明显大于旧图片0.12）；留白不足或完整周期边界受限时缩小并记录告警，显式尺寸覆盖默认偏好。
- 自动落点使用支撑线、下半部留白和本地经典Haar/HOG保护检测；排除上半部支撑候选，检测可能漏检/误检，不承诺理解车顶、桌面等语义或实现真实遮挡。
- 素材从原始帧按原生像素重建，不再默认压到384高；预乘Alpha高质量采样减少缩放边缘污染，不生成缺失细节，也不宣称凭空高清恢复。
- 现阶段提供附件输入和脚本合成，没有独立的拖拽上传网页。

## 执行入口

固定入口为 [合成脚本](scripts/render_overlay.py)，其内部使用 [合成核心模块](scripts/overlay_core.py)。素材维护使用 [预处理脚本](scripts/prepare_assets.py)；回归测试由 [测试脚本](scripts/test_overlay.py) 调用 [测试输入生成器](scripts/make_test_inputs.py)，不需要在正常任务中运行测试输入生成器。

从本 Skill 根目录执行；附件使用本次请求真实存在的文件路径，不复用历史会话的临时路径。

1. 检查 `ffmpeg`、`ffprobe` 和 Python 包。缺 Python 包时执行：

```bash
python3 -m pip install -r scripts/requirements.txt
```

FFmpeg 需系统提供，缺失时明确报告，不把缺工具误报为渲染成功。

2. 确认 `assets/dog_manifest.json` 和其引用的透明缓存存在。完整包已包含处理后的缓存，常规运行不需要重新抠绿。

3. 对用户提供的真实附件直接调用固定入口：

```bash
python3 scripts/render_overlay.py --inputs '/path/photo.jpg' '/path/video.mp4' --output-dir temp_data/output --seed 42
```

指定明确的输出目录，返回命令生成的实际 MP4 链接。批次状态、镜头计划、告警、耗时从该次 JSON 结果读取；部分失败时仍交付成功文件并解释失败，不宣称全部成功。

## 常见调整

先用 `python3 scripts/render_overlay.py --help` 检查当前参数；不猜不存在的选项。

```bash
# 图片片长与高清模式
python3 scripts/render_overlay.py --inputs '/path/photo.png' --output-dir temp_data/hd --image-duration 5 --quality hd --seed 42

# 人工指定视频切点，单位为秒
python3 scripts/render_overlay.py --inputs '/path/video.mp4' --output-dir temp_data/cuts --cuts '1.5,3.2,5.0' --seed 42

# 指定脚底锚点与尺寸：坐标归一化，左上为(0,0)，右下为(1,1)
python3 scripts/render_overlay.py --inputs '/path/photo.jpg' --output-dir temp_data/manual --anchor 0.25 0.72 --height-ratio 0.12 --seed 42

# 仅分析选位并输出计划，不渲染视频
python3 scripts/render_overlay.py --inputs '/path/video.mp4' --output-dir temp_data/plan --plan-only --seed 42
```

重新随机时改变 seed；复现时保留相同输入、参数、素材版本和 seed。手动切点只用于对应视频，不把一组切点默认套在长度不一致的批次。

`--dog-id` 只指定第一镜头的狗，后续镜头仍使用不相邻重复的洗牌袋。首版仅支持 `--tracking off`；`flow` 会明确报错，不能把它当作已实现的移动跟踪。

## 素材维护

运行时不下载线上素材。仅在主动维护素材库或缓存丢失时运行：

```bash
python3 scripts/prepare_assets.py --source-dir assets/source --output-dir assets
```

若重建缓存，检查棋盘预览和整个周期的黑轮廓、绿边、耳朵/尾巴裁切和脚底稳定性。透明效果是离线处理后的结果；输入源 MP4 本身没有 Alpha。素材版本与 sha256 来自清单，不因看起来相似而混用缓存。

## 交付检查

- 输出文件存在且可被 ffprobe 读取，画幅、时长、音轨符合本次参数。
- 镜头计划里每镜头一个 `dog_id`，相邻狗不同；切点附近没有额外叠加分支造成双狗。
- 尺寸约束针对完整动作周期，不只针对第一帧。
- 明确说明告警、退化选位、非生成式启发式的限制，不声称语义理解或真实三维跟踪。
- 只报告实测耗时，区分素材准备、分析、编码和文件传输。

运行回归：

```bash
python3 scripts/test_overlay.py
```

详细参数、测试基准与首版边界见 [使用与验收](references/usage-and-validation.md)。素材来源与处理约定见 [素材来源](references/asset-provenance.md)。
