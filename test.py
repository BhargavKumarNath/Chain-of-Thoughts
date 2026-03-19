import urllib.request
import json

def test_query(q):
    req = urllib.request.Request(
        'http://localhost:8000/api/v1/reason', 
        data=json.dumps({'query': q}).encode('utf-8'), 
        headers={'Content-Type': 'application/json'}
    )
    res = urllib.request.urlopen(req).read().decode('utf-8')
    data = json.loads(res)
    return {
        "query": q,
        "strategy": data['strategy_selected'],
        "hallucination_risk": data['hallucination_risk'],
        "latency": data['latency_ms'],
        "final_answer": data['final_answer']
    }

results = [
    test_query("What is 1+1?"),
    test_query("I need to understand the implications of quantum tunneling on modern semiconductor architecture, specifically referencing sub-2nm node constraints and potential material alternatives like graphene. Also, how does this relate to Moore's Law?")
]

with open("result.json", "w") as f:
    json.dump(results, f, indent=2)
