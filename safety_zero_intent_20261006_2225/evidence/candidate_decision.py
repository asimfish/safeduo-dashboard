"""Conservative descriptive decision, frozen before new policy outcomes."""
def decide(result):
    t={r['mode']:r for r in result['totals']};a,b=t['joint_reference'],t['zero_inclusive']
    paired=[r for r in result['paired'] if r['a']=='joint_reference' and r['b']=='zero_inclusive']
    assert len(paired)==3
    reasons=[]
    if sum(r['new_failures'] for r in paired)>0:reasons.append('新增配对严格失败')
    if b['violations']>a['violations']:reasons.append('严格失败数量增加')
    if b['deep']>a['deep']:reasons.append('深度失败数量增加')
    if b['deep_env_steps']>a['deep_env_steps']:reasons.append('深度违规环境帧增加')
    if reasons:return dict(code='REJECTED_ADVERSE_ENDPOINTS',text='候选出现不利安全端点，拒绝采用',reasons=reasons)
    if b['violations']:return dict(code='RESEARCH_ONLY_RESIDUAL_FAILURES',text='仍有失败，仅保留研究验证',reasons=['存在残余严格失败'])
    return dict(code='OBSERVED_ZERO_FAILURES_LIMITED_SCOPE_ONLY',text='本次候选观察到零失败，仅限本实验范围',reasons=[])
