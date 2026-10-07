# 本轮执行检查点（未交付）

2026-10-07 23:43北京时间。用户要求继续推进更完整实验和验收。唯一活跃原生进程PID786653、shell786650、exec session32719，12env开发120档/118独特目标、360cycle、720U手路径、21600step预注册259200envstates。原生自碰开启、GPU0/camera0（两卡可见），GPU1有无关chembench进程不要动。开发已14640step/175680envstates，96/144图；进度详见NASdevelopment/progress.json。循环没有q/qd/物体位姿重置。等待真正退出，不能提前叫完成。

本目录 native_probe.py、joint_paths.py、analyze.py 和源/资产303条在REGISTRATION_DEVELOPMENT.json已锁，禁止改写。修改新版本另存；本轮源hash检查没有漂移。已15软件准入合同测试通过，RED保留0.1 float32边界修复；check_paths6480纯命令检查通过，不是物理样本。初步原对照全伙伴法向max232.5374N，open和.45max0；非完整评分。独立目标维度thumb1/2和耦合finger，全部grid0..1，20IID+20LHS新OSseed；机械臂/物体不随机。

开发完成：poll session32719确认exit0；记录DEVELOPMENT_EXIT.json（不要以后改写已登记json）。
python analyze.py --registration REGISTRATION_DEVELOPMENT.json --raw /mnt/nas/data/lyf/double_hand/safety_hand_closure_qualification_20261007/development --out DEVELOPMENT_RESULTS.json
必须先看到完整结果。scorer原字段unique_profiles指槽位120，真实独特118；COUNT_FIELD_NOTE保留此标签问题。score和脚本不调用候选生成器。

然后 register_validation.py（源代码都已准备）按事前OS validationseed无替换从6路径全部通过精确目标选前12，移除重复open；3新ref2/8/20。冻结PASSPORT和REGISTRATION_VALIDATION/VALIDATION_SEAL。native_validation.py生成264图、连杆pose、4连续video视角60帧，在每步真实nativeq/qd/自碰normal下接入HandAdmission。32未知随机只请求拒绝不执行，对原不合格目标也请求拒绝；未执行目标不得算物理安全样本。原生请求日志含32逐项理由。运行命令同开发GPU/env，--out NAS/validation；必须新目录。

验证完成记录VALIDATION_EXIT.json，analyze.py生成VALIDATION_RESULTS.json，audit_admission.py独立从前一物理步tensor验证申请。

补同初态配对：register_paired.py生成6env/720step，3ref×bypass/guard；两个U手共12handpaths。native_paired.py+paired_paths.py，reg REGISTRATION_PAIRED.json，--out NAS/paired。guard拒绝同旧目标、始终开放；bypass直接闭合原.75，后返回。PAIR_CONTRACT已事前冻结（开发评分前）。新增初始化objects原生读回：q/qd/target配对bitexact、物体局部位置≤1e-6m/速度bitexact。capture：两组ref2的60detail连续帧＋3ref各closed/open四视角，168原图。analyze_paired.py生成PAIRED_RESULTS.json独立评分，保存PAIRED_EXIT.json。guard任务成功=0，不能宣称System0臂投影或真实抓持收益。

所有原生运行：CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=本目录:/home/liyufeng/safeduo/src LD_PRELOAD=/home/liyufeng/miniforge3/envs/safeduo/lib/libstdc++.so.6 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y OMP/MKL/OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 /home/liyufeng/miniforge3/envs/safeduo/bin/python <native script> --registration <reg> --out <新NAS目录> --headless --device cuda:0 --kit_args '--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false' > <run.log> 2>&1。Kit可能错误仍exit0，必须receipt/no failure/source SHA/independent full score。constructor~150s材质网络timeout是已知，日志保留。native类已camera pop before env.reset，SimulationContext.clear_instance beforeclose。

后处理源encode_videos.py（6视频：验证top/front/side/detail + paired unsafe_detail/guard_detail，各60原PNG 10fps），build_payload.py（全部3run raw/results/曲线、force每12步max，角度窗口末native样本；OSseed字符串展示防JS精度丢失），build_panel.py、check_browser.py已备。未在freeze后编辑已有登记.py/.md/.json，须保持所有source_sha256稳定。后新增文件可以。

发布：主树 /home/liyufeng/safeduo-dashboard-benchmark-protocol-20261006 clean exp/hand-initialization-20261007 HEAD/publicmain f8f18455。剩余仅178152字节到1GiB，payload/全部media放归档，主页面最多~40KB。主branch unprotected false已API查；push前再fetch检查并保留远端并发修改。用户已授权公开主面板/视频，无需再问。
归档树 /home/liyufeng/safeduo-dashboard-object-media-20261006 clean exp/hand-initialization-media-20261007 HEADd7e7ef18；新exp/hand-closure-media-20261007分支从d7起，不覆盖旧证据；拷本目录稳定科研资料+NAS三个raw完整+6video（不要videos内临时symlink frame副本）。建MEDIA_MANIFEST files/file/sha256/bytes，copy到archive后本地验证，然后显式gitstage/commit/push新branch。
verify_archive.py --commit NEW --local archivedir --mirror NAS/<unique archive mirror> --out ARCHIVE_DOWNLOAD.json：下载每个固定raw文件逐字节SHA比较，保留镜像；不必下载整个历史1GBZIP，必须说明逐文件验收而非声称ZIP已下载。
build_panel.py --archive-commit NEW --repo 主树（新doc safety_hand_closure_qualification_20261007，root增加handClosureEvidence，旧root去除card byteexact）。固定SHA externalGZIP fetch，CORS验证。主树少量5files+root。旧预览ownedPID2112139 serve主树8973仍可用，rangepreview支持video206。浏览器：PLAYWRIGHT_BROWSERS_PATH=/tmp/safeduo_dashboard_browser PYTHONPATH=/tmp/safeduo_dashboard_browser_tools /usr/bin/python3 check_browser.py --base http://127.0.0.1:8973 --repo 主树 --out 新dir。浏览器desktop/mobile全部792dev+validation曲线、75matrixcells、6pairedcurves、全部576原图、6VP9video实际播放，既有cards完整。看near原图和desktop截图，不做imagegen/图像编辑。

完成源码/资产前后SHA、数据/准入独立评分、公版媒体下载完整、局部和线上browser journey、public精确files+exactPagescommit（API publicrequests，无gh命令）。Git只exp分支提交，从exp推未protectedmain（已授权）。Main父root旧所有card保持byteexact，archive旧scientificdata原字节完整。最终DELIVERY_CLOSURE、本地终验/网页链接，明确真实grasp/release/构造期测力/六臂对全操作空间仍未完成。

重要：不能派子代理；未创建goal；permission never，不传sandbox_permissions。每60秒须用户commentary。原Any历史接口旧turn4010messages，这轮不冒称有新历史。不可把手部准入拒绝的空手无接触叫完整机器人安全或抓持成功。

## 2026-10-08 00:05追加：逐点法向审计
开发未结束，已16560step/198720envstates。发现zero-vector对照仍有internal contact count（open8/4,safe45 16/16），可能零力近邻，不知是否抵消。开发旧raw只存count不存pointforce，不能假称有scalar qualification。新SCALAR_NORMAL_CONTRACT.md在验证/配对运行之前冻结。contact_points.py sum(abs(force)) eachsensor/filter，capacity/start-range/finiteness校验，只读有效count范围，4contracttests PASS，不是physics。native_validation/native_paired与runtimegate改用fresh scalar totals，仍存vector/count/nativeq等。
audit_admission.py也按previous-step scalar totals复算。
必须在validation和paired完整且旧analyze两个RESULTS输出后，运行audit_scalar_contacts.py，输出VALIDATION_FULL_RESULTS.json、PAIRED_FULL_RESULTS.json以及两个SCALAR_AUDIT。trianglecheck vector_norm≤scalar+.001+2e-6scalar；scalar≤.1N才qualified。旧RESULTS不修改。build_payload已改用FULL结果、validation/paired curves用scalar，dev仍vector并声明无法排除抵消。build_panel和UI同初态force显示scalar。source code已py_compile，新计数说明120槽/118specs。
执行后处理顺序：三run终止→旧analyze+audit_admission→audit_scalar_contacts→encode_videos→build_payload→acceptance_matrix→归档manifest/public发布。build_payload的links含ACCEPTANCE_MATRIX；这文件生成后归档即可，非native输入，不影响已封存source。
新的288份源码snapshot在source_snapshot/src，SOURCE_SNAPSHOT.json字节匹配原注册；USD只hash未复制，嵌套remote textures未全封存。将snapshot一并归档、secretpattern只文件名扫描无匹配。
