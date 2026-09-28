import subprocess
import time
import csv
from datetime import datetime

CONTEXT = "kind-iaha-postgres"
NAMESPACE = "default"
RESULTS_FILE = "rto_results.csv"

def run_kubectl(cmd):
    result = subprocess.run(
        f"kubectl {cmd} --context={CONTEXT}",
        shell=True, capture_output=True, text=True
    )
    return result.stdout.strip()

def get_primary():
    out = run_kubectl("get cluster iaha-pg -o jsonpath='{.status.currentPrimary}'")
    return out.replace("'", "")

def get_cluster_state():
    out = run_kubectl("get cluster iaha-pg -o jsonpath='{.status.phase}'")
    return out.replace("'", "")

def wait_for_healthy(timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        state = get_cluster_state()
        if "healthy" in state.lower():
            return time.time() - start
        time.sleep(1)
    return -1

def run_failover_test(test_num):
    print(f"\n--- Test {test_num}/30 ---")
    
    # Identifier le primary actuel
    primary = get_primary()
    print(f"Primary avant failover : {primary}")
    
    # Attendre que le cluster soit stable
    time.sleep(5)
    
    # Déclencher le failover
    t_start = time.time()
    run_kubectl(f"delete pod {primary} -n {NAMESPACE}")
    print(f"Pod {primary} supprimé à {datetime.now().strftime('%H:%M:%S')}")
    
    # Attendre le nouveau primary
    time.sleep(3)
    new_primary = None
    while not new_primary or new_primary == primary:
        new_primary = get_primary()
        time.sleep(1)
    
    t_promotion = time.time() - t_start
    print(f"Nouveau primary : {new_primary} en {t_promotion:.2f}s")
    
    # Attendre healthy state complet
    rto = wait_for_healthy()
    print(f"Cluster healthy en {rto:.2f}s")
    
    # Attendre stabilisation avant prochain test
    time.sleep(30)
    
    return {
        "test": test_num,
        "former_primary": primary,
        "new_primary": new_primary,
        "promotion_time_s": round(t_promotion, 2),
        "rto_s": round(rto, 2),
        "timestamp": datetime.now().isoformat()
    }

def main():
    print("=== IAHA-X Failover Test Suite ===")
    print(f"Cluster: {CONTEXT}")
    print(f"Résultats: {RESULTS_FILE}")
    
    results = []
    
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "test", "former_primary", "new_primary",
            "promotion_time_s", "rto_s", "timestamp"
        ])
        writer.writeheader()
        
        for i in range(1, 31):
            try:
                result = run_failover_test(i)
                results.append(result)
                writer.writerow(result)
                f.flush()
                print(f"✓ Test {i} enregistré — RTO: {result['rto_s']}s")
            except Exception as e:
                print(f"✗ Test {i} échoué : {e}")
    
    # Statistiques finales
    rtos = [r["rto_s"] for r in results if r["rto_s"] > 0]
    if rtos:
        import statistics
        print(f"\n=== Résultats finaux ({len(rtos)} tests) ===")
        print(f"RTO moyen  : {statistics.mean(rtos):.2f}s")
        print(f"RTO médian : {statistics.median(rtos):.2f}s")
        print(f"RTO min    : {min(rtos):.2f}s")
        print(f"RTO max    : {max(rtos):.2f}s")
        print(f"Écart-type : {statistics.stdev(rtos):.2f}s")

if __name__ == "__main__":
    main()