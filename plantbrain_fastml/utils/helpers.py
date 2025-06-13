from joblib import cpu_count

def get_effective_n_jobs(n_jobs):
    if n_jobs is None:
        return 1
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    return n_jobs