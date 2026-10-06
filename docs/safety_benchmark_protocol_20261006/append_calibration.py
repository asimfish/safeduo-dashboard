"""Publish the bounded native calibration without changing final-trial denominators."""
from pathlib import Path
import argparse,json,hashlib,shutil,gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();doc=a.repo/'docs'/HERE.name;d=doc/'native_calibration';d.mkdir(exist_ok=True)
    receipt=json.loads((HERE/'NATIVE_CALIBRATION_CLOSURE.json').read_text());x=receipt['result']
    assert receipt['status']=='PASS_CLOSED_PRIMITIVE_MEASUREMENT_CONTROLS' and receipt['process_exit_code']==0 and receipt['full_G0_pass'] is False
    raw=Path('/mnt/nas/data/lyf/double_hand/safety_benchmark_protocol_20261006/native_calibration_3/native_trace.npz')
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==x['trace_sha256']
    shutil.copy2(raw,d/'native_trace.npz')
    names=['NATIVE_CALIBRATION_REGISTRATION.json','NATIVE_CALIBRATION_REGISTRATION_V2.json','NATIVE_CALIBRATION_REGISTRATION_V3.json',
        'NATIVE_CALIBRATION_FIRST_RUNTIME.json','NATIVE_SHUTDOWN_DIAGNOSIS.json','NATIVE_CALIBRATION_CLOSURE.json',
        'native_contact_calibration.py','native_contact_calibration_v2.py','native_contact_calibration_v3.py',
        'native_calibration_1.log','native_calibration_2.log','native_calibration_3.log']
    logs={}
    for n in names:
        if n.endswith('.log'):
            blob=(HERE/n).read_bytes();(d/(n+'.gz')).write_bytes(gzip.compress(blob,mtime=0))
            logs[n]=dict(raw_sha256=hashlib.sha256(blob).hexdigest(),public_gzip=n+'.gz')
        else:shutil.copy2(HERE/n,d/n)
    (d/'LOG_MANIFEST.json').write_text(json.dumps(logs,indent=2)+'\n')
    shutil.copy2(HERE/'append_calibration.py',doc/'append_calibration.py')
    with np.load(raw,allow_pickle=False) as z:pos=z['positions'].copy();forces=z['force_matrix'].copy()
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');t=(np.arange(360)+1)*x['dt_s']
    for i,mass in enumerate(x['registered_masses_kg']):
        axes[0].plot(t,pos[:,i,2],label=f'{mass}kg')
        axes[1].plot(t,np.abs(forces[:,i,i,2]),label=f'{mass}kg')
        axes[1].axhline(mass*9.81,linestyle=':',alpha=.45,color=f'C{i}')
    axes[0].axhline(.1,color='black',linestyle=':',label='Analytic rest height')
    axes[0].set(xlabel='Native physics time (s)',ylabel='Cube center height (m)',title='Native state: free fall and rest')
    axes[1].set(xlabel='Native physics time (s)',ylabel='Own-support vertical force magnitude (N)',title='Attributed support force; dotted=mass*g')
    for axis in axes:axis.legend(fontsize=8)
    fig.suptitle('Controlled primitives only; no four-arm, grasp or hardware acceptance')
    for ext in ('png','pdf','svg'):
        target=doc/'figures'/('native_contact_calibration.'+ext);fig.savefig(target,dpi=180,bbox_inches='tight')
        if ext=='svg':target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)
    data=json.loads((doc/'data.json').read_text());data['native_calibration']=receipt
    (doc/'data.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n')
    rows=''.join(f'<tr><td>{m:.1f}kg</td><td>{w:.6f}N</td><td>{f:.6f}N</td><td>{err*100:.6f}%</td></tr>' for m,w,f,err in zip(x['registered_masses_kg'],x['expected_weight_n'],x['settled_median_abs_vertical_n'],x['relative_weight_error']))
    section='''<section id="nativeCalibration"><h2>已执行：受控原生接触基础校准</h2><p>三种已知质量的立方体自由下落至各自支持面；每个物体同时过滤三个支持面及一个远处的错误对象。360个物理步、每步1/120秒，原生力张量设备cuda:1。分离时、其它支持面及错误对象的报告力均为0。</p><div class="table"><table><thead><tr><th>质量</th><th>理论自重</th><th>静置力中位数</th><th>相对误差</th></tr></thead><tbody id="nativeRows">'''+rows+'''</tbody></table></div><figure><img src="figures/native_contact_calibration.png" alt="三种已知质量物体的自由下落原生高度与支持力"><figcaption>原生状态与逐端点支持力；接触瞬间冲击峰保留。<a href="figures/native_contact_calibration.pdf">PDF</a> · <a href="figures/native_contact_calibration.svg">SVG</a></figcaption></figure><p>逐接触法向力聚合与原生力矩阵的最大分量绝对误差约0.0000305N，静置高度与解析值一致，接触缓冲区没有达到容量。完整数据已离线检查有限值、归因零对照、静置自重及SHA。</p><p>第一轮设备枚举问题、第二轮关闭停滞均保留。停止回调在时间轴停止后持续渲染是安装版本中的关闭问题；第三轮在关闭前调用公开clear_instance清理接口，正常退出0，且与第二轮物理轨迹逐位相同。模拟器源码与既有控制器未改动。</p><div class="warning">这是G0的一项基础测量证据。三种质量不是三个独立总体安全试验，1080个物体状态也不是1080次独立试验；没有据此放行四臂全场景、真实夹持或最终20类矩阵。</div><div class="downloads"><a href="native_calibration/NATIVE_CALIBRATION_CLOSURE.json">关闭与数据复核凭证</a><a href="native_calibration/native_trace.npz">完整原生轨迹</a><a href="native_calibration/NATIVE_CALIBRATION_REGISTRATION_V3.json">执行前冻结参数</a><a href="native_calibration/native_contact_calibration_v3.py">可复跑脚本</a><a href="native_calibration/NATIVE_SHUTDOWN_DIAGNOSIS.json">关闭原因与修复</a></div></section>'''
    html=(doc/'index.html').read_text();marker='<section id="nativeCalibration">'
    if marker in html:
        import re;html=re.sub(r'<section id="nativeCalibration">.*?</section>',section,html,flags=re.S)
    else:html=html.replace('<section><h2>保留之前的真实物体实验</h2>',section+'\n<section><h2>保留之前的真实物体实验</h2>',1)
    html=html.replace('本页新增的是封存数据的覆盖审计和下一阶段的实验规范。','本页包含封存数据覆盖审计、受控原生接触基础校准及下一阶段的实验规范。')
    html=html.replace('新协议物理窗口','最终矩阵物理窗口')
    (doc/'index.html').write_text(html)
    root=a.repo/'index.html';text=root.read_text();text=text.replace('新协议物理试验尚未执行。六类随机','正式矩阵物理试验尚未执行；原生接触基础校准已完成。六类随机')
    root.write_text(text);print('APPENDED_NATIVE_CONTROLS',x['trace_sha256'])
if __name__=='__main__':main()
