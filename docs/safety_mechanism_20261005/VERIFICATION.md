# 交付检查（独立于安全机制准入）

交付要求：固定三新种子、四方法同外部输入和初态、各进程完整960步，共768窗口；全部失败保留，解析FIFO、目标增量/软限位/包络契约通过；源码与actor/参数hash不变。原64因果重播与128开发窗口另列，不计新增独立轨迹。保留所有旧scientific字段、曲线、视频；公开面板显示最新结果及操作范围，移动宽度390px无页面溢出。

变更仅独立实验目录、dashboard/index.html与docs、安全数据新safety_mechanism字段；没有生产actor或保护器变更。安全准入需要零违规且其他任务独立验证，不受“数据交付通过”替代。

检查：完整analyze_mechanism（源/配置/hash、初态引用、无future-policy选择、输入/FIFO逐帧、全部官方裕量、独立包络oracle）；4项CPU RED/GREEN契约；Python compileall；Git显式diff检查；准确发布目录敏感文本与禁止NPZ/PT扫描；真实Chromium逐个检查新/旧覆盖选项和曲线、表格数字、JS错误、截图；HTTP200与发布JSON/REPORT相等。

没有依赖或登录边界变更，不主张新增威胁模型或性能提升。本静态页无package/CI测试契约；GitHub部署与公开实际页面验证分开记录，不冒称远端测试已通过。computer-use技能目录未提供，已使用现有Playwright驱动实际Chromium取得页面状态与视觉证据；无需安装依赖。旧科研合成/冻结实验不重复跑，核验原完整数据字段及可交互页面回归。
