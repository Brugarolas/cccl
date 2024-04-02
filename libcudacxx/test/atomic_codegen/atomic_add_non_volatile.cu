#include <cuda/atomic>

__global__ void add_relaxed_device_non_volatile(int* data, int n) {
    if (n) {
        auto ref = cuda::atomic_ref<int, cuda::thread_scope_device>{*(data)};
        ref.fetch_add(n, cuda::std::memory_order_relaxed);
    }
}

/*
## SM80 checks
; SM8X: Fatbin elf code
; SM8X: code for sm_8{{[0-9]}}
; SM8X-DAG:  {{^.*}}Function : {{.*}}add_relaxed_device_non_volatile{{.*$}}
; SM8X-NOT:  {{^.*}}STL.64{{.*$}}
; SM8X:      {{^.*}}ATOM.E.ADD.STRONG.GPU{{.*$}}
; SM8X-DAG:  {{^.*}}EXIT{{.*$}}
*/
