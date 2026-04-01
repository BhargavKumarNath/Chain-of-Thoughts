import urllib.request
import json
import time

def post_query(q):
    req = urllib.request.Request('http://localhost:8000/api/v1/reason', data=json.dumps({'query': q}).encode(), headers={'Content-Type': 'application/json'})
    try:
        r = urllib.request.urlopen(req)
        return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {'error': e.read()}

print('1. Testing Instruction Completion with: explain why the sky is blue')
res1 = post_query('explain why the sky is blue')
print(f"Strategy: {res1.get('strategy_selected')} Completion Score: {res1.get('completion_score')} Trust: {res1.get('trust_score', {}).get('aggregate_score')}")

print('2. Testing Confidence Inversion Guard: solve this paradox: if Pinocchio says his nose will grow, will it?')
res2 = post_query('solve this paradox: if Pinocchio says his nose will grow, will it?')
print(f"Strategy: {res2.get('strategy_selected')} Conf: {res2.get('verification_confidence')} Level: {res2.get('difficulty_level')} Details: {[d['method'] for d in res2.get('verification_details', [])]}")

print('3. Testing optimize policy')
req_opt = urllib.request.Request('http://localhost:8000/api/v1/policy/optimize', data=b'', headers={'Content-Type': 'application/json'}, method='POST')
opt_res = json.loads(urllib.request.urlopen(req_opt).read())
print('Policy Weights:', json.dumps(opt_res['policy']['weights'], indent=2))
