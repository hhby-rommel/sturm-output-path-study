# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Guard simulations and one historical-engine numeric bridge; no formal/test run."""
from pathlib import Path
import sys,json,hashlib,tempfile,types,copy,argparse,importlib
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import torch
from scripts import decoder_attribution_train_val_20261003_v1 as runner
from scripts import decoder_attribution_serial_20261003_v1 as serial
from src.training.independent_decoder_preflight import rng_state,restore_rng,compare_states
from src.training.single_sensor_decoder_control import LocalSelectionState
from src.training.masked_segmentation import MaskedSemanticLoss
from src.training.engine import train_one_epoch
def write(p,x):p.write_text(json.dumps(x,indent=2))
def expect_error(fn,expected):
 try:fn()
 except expected:return True
 raise AssertionError('guard did not reject')
def main():
 p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();out=a.output_dir.resolve();out.mkdir(exist_ok=False,parents=True);lock=runner.read(a.lock);results={};torch.set_num_threads(4)
 results['source_asset_gate']=runner.verify(R,lock) is None
 selector=LocalSelectionState()
 def val(d,c):return {'clean':{'overall':{'flood_f1':c}},'degraded':{'overall':{'flood_f1':d}}}
 selector.observe(1,val(.7,.8));selector.observe(2,val(.7,.8));assert selector.best['epoch']==1
 selector.observe(3,val(.7,.81));assert selector.best['epoch']==3 and selector.stale==0
 for i in range(4,13):assert not selector.observe(i,val(.69,.9))
 assert selector.observe(13,val(.69,.9));results['tie_clean_earlier_and_patience10']=True
 args=argparse.Namespace(phase='formal',arm='no_input',seed=2026,project_root=R,output_dir=out/'never_formal',lock=a.lock,approval=None,health_receipt=None,device='cuda')
 results['missing_authorization_rejected']=expect_error(lambda:runner.run(args,lock),PermissionError)
 assert not args.output_dir.exists()
 cpu_receipt=out/'cpu_receipt.json';write(cpu_receipt,{'status':'INDEPENDENT_LOCAL_CPU_ALL_FOUR_ARM_HEALTH_PASSED','lock_sha256':runner.sha(a.lock)})
 approval=out/'SIMULATION_ONLY_NOT_LAUNCH_AUTHORIZATION.json';tasks=lock['formal_order'];write(approval,{'approved':True,'action':'train_val','lock_sha256':runner.sha(a.lock),'tasks':tasks,'scope':'unit simulation only; no real subprocess launched'})
 args.approval=approval;args.health_receipt=cpu_receipt
 results['CPU_receipt_not_GPU_admission']=expect_error(lambda:runner.run(args,lock),PermissionError)
 args.phase='health';args.arm='full';args.seed=10501;args.device='cpu';args.output_dir=out/'existing';args.output_dir.mkdir();(args.output_dir/'sentinel.txt').write_text('untouched')
 before=runner.sha(args.output_dir/'sentinel.txt');results['existing_directory_refused']=expect_error(lambda:runner.run(args,lock),FileExistsError);assert runner.sha(args.output_dir/'sentinel.txt')==before and list(args.output_dir.iterdir())==[args.output_dir/'sentinel.txt']
 # Pure controller simulation: fake receipt and mocked process call are never launch files.
 simulated=out/'SIMULATION_GPU_RECEIPT_NOT_VALID.json';write(simulated,{'status':'SERVER_GPU_ALL_FOUR_ARM_HEALTH_PASSED','lock_sha256':runner.sha(a.lock),'scope':'mock only'})
 argv=['serial','--project-root',str(R),'--output-root',str(out/'serial_simulation'),'--lock',str(a.lock),'--approval',str(approval),'--health-receipt',str(simulated)]
 with patch.object(sys,'argv',argv),patch.object(serial.subprocess,'call',return_value=7) as call:
  results['serial_first_failure_stops']=expect_error(serial.main,RuntimeError);assert call.call_count==1
  cmd=call.call_args.args[0];assert '--phase' in cmd and cmd[cmd.index('--phase')+1]=='formal' and '--resume' not in cmd and '--test' not in cmd
 results['no_automatic_test_or_resume']=True
 # Load the historical committed engine from the frozen bundle, not an internal cache.
 historic=importlib.import_module('src.training.historical_engine_oracle')
 model,init=runner.build('full',10501,R,torch.device('cpu'));other=copy.deepcopy(model);criterion=MaskedSemanticLoss(aux_weight=.4)
 loader,_=runner.loader_pair(R,10501,runner.read(R/'health_fixture.json'));groups=next(iter(loader));assert set(groups)=={'S1','S2'}
 opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);oo=torch.optim.AdamW(other.parameters(),lr=.0003,weight_decay=.0001);state=rng_state()
 restore_rng(state);new=train_one_epoch(model,[groups],criterion,opt,device=torch.device('cpu'),model_variant='m1_degraded',grad_clip_norm=5,freeze_batchnorm=True)
 restore_rng(state);old=historic.train_one_epoch(other,[groups],criterion,oo,device=torch.device('cpu'),model_variant='m1_degraded',grad_clip_norm=5,freeze_batchnorm=True)
 compare_states(model.state_dict(),other.state_dict());compare_states(opt.state_dict(),oo.state_dict())
 results['original_engine_next_update_bridge']=dict(tolerance_pass=True,new_loss=new['loss'],historical_loss=old['loss'],atol=1e-6,rtol=1e-5,extra_updates=2,scope='one mixed-sensor real fixture update; not all-history replay')
 write(out/'guard_and_bridge_receipt.json',dict(status='GUARDS_AND_HISTORICAL_ENGINE_FIXTURE_BRIDGE_PASSED',checks=results,formal_training_started=False,test_enabled=False))
 print('GUARDS_AND_ENGINE_BRIDGE_PASSED',flush=True)
if __name__=='__main__':main()
