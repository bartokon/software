NVCC = /usr/local/cuda-11.8/bin/nvcc
HIPIFY = /opt/rocm-6.0.0/bin/hipify-clang
HIPCC = /opt/rocm-6.0.0/bin/hipcc
DPCT = /opt/intel/oneapi/2024.0/bin/dpct
#DPCC = /opt/intel/oneapi/2024.0/bin/dpcpp
DPCC = /opt/intel/oneapi/2024.0/bin/icpx -fsycl

#CPP
SRC_DIR=src
CPP_SRC_FILES=$(wildcard $(SRC_DIR)/*.cpp)
HPP_SRC_FILES=$(wildcard $(SRC_DIR)/*.hpp)
CUDA_SRC_FILES=$(wildcard $(SRC_DIR)/*.cu)

#CUDA
CUDA_OBJ_DIR=cuda
CUDA_OBJ_FILES=$(patsubst $(SRC_DIR)/%.cu,$(CUDA_OBJ_DIR)/%.o,$(CUDA_SRC_FILES))
CUDA_OBJ_FILES+=$(patsubst $(SRC_DIR)/%.cpp,$(CUDA_OBJ_DIR)/%.o,$(CPP_SRC_FILES))

#HIP
HIP_OBJ_DIR=hip
HIP_OBJ_FILES=$(patsubst $(SRC_DIR)/%.cu,$(HIP_OBJ_DIR)/%.o,$(CUDA_SRC_FILES))
HIP_OBJ_FILES+=$(patsubst $(SRC_DIR)/%.cpp,$(HIP_OBJ_DIR)/%.o,$(CPP_SRC_FILES))

#SYCL
DPC_OBJ_DIR=dpc
DPC_OBJ_FILES=$(patsubst $(SRC_DIR)/%.cu,$(DPC_OBJ_DIR)/%.o,$(CUDA_SRC_FILES))
DPC_OBJ_FILES+=$(patsubst $(SRC_DIR)/%.cpp,$(DPC_OBJ_DIR)/%.o,$(CPP_SRC_FILES))

.PHONY: info
info:
	@echo "CPP_SRC_FILES: $(CPP_SRC_FILES)"
	@echo "CUDA_SRC_FILES: $(CUDA_SRC_FILES)"
	@echo "CUDA_OBJ_FILES: $(CUDA_OBJ_FILES)"
	@echo "HIP_OBJ_FILES: $(HIP_OBJ_FILES)"
	@echo "DPC_OBJ_FILES: $(DPC_OBJ_FILES)"

all: cuda_app.elf hip_app.elf sycl_app.elf
	@echo "All done."

#CUDA
$(CUDA_OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "NVCC compile: $<"
	$(NVCC) -c -o $@ $<

$(CUDA_OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "NVCC compile: $<"
	$(NVCC) -c -o $@ $<

cuda_app.elf: $(CUDA_OBJ_FILES)
	@echo "Linking CUDA executable..."
	$(NVCC) $(CUDA_OBJ_FILES) -L$(CUDA_DIR)/lib64 -lcudart -o cuda_app.elf

#HIP TRANSLATE
.PRECIOUS: $(HIP_OBJ_DIR)/%.cu.hip
$(HIP_OBJ_DIR)/%.cu.hip: $(SRC_DIR)/%.cu
	@echo "HIP translate $<"
	$(HIPIFY) $< --o-dir=$(HIP_OBJ_DIR)

$(HIP_OBJ_DIR)/%.o: $(HIP_OBJ_DIR)/%.cu.hip
	@echo "HIP compile: $<"
	$(HIPCC) -c -o $@ $<

$(HIP_OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "HIP compile: $<"
	$(HIPCC) -c -o $@ $<

hip_app.elf: $(HIP_OBJ_FILES)
	@echo "Linking HIP executable..."
	$(HIPCC) $(HIP_OBJ_FILES) -o hip_app.elf

#DPC TRANSLATE
.PRECIOUS: $(DPC_OBJ_DIR)/%.dp.cpp
$(DPC_OBJ_DIR)/%.dp.cpp: $(SRC_DIR)/%.cu
	@echo "DPCT translate $<"
	$(DPCT) $< --out-root=$(DPC_OBJ_DIR)

$(DPC_OBJ_DIR)/%.o: $(DPC_OBJ_DIR)/%.dp.cpp
	@echo "DPC compile $<"
	$(DPCC) -c -I/opt/intel/oneapi/2024.0/include/ -o $@ $<

$(DPC_OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "DPCC compile $<"
	$(DPCC) -c -o $@ $<

sycl_app.elf: $(DPC_OBJ_FILES)
	@echo "Linking DPCC executable..."
#	$(shell source /opt/intel/oneapi/2024.0/oneapi-vars.sh)
	$(DPCC) $(DPC_OBJ_FILES) -o sycl_app.elf -v
#	$(DPCC) $(DPC_OBJ_FILES) -L/opt/intel/oneapi/2024.0/lib/ -o sycl_app.elf -v
#	$(DPCC) $(DPC_OBJ_FILES) -Wl,-rpath,/opt/intel/oneapi/2024.0/lib/,/opt/intel/oneapi/2024.0/lib32/ -lonnxruntime -o sycl_app.elf -v


clean:
	@echo "Cleaning up files..."
	rm -f $(CUDA_OBJ_DIR)/*
	rm -f $(HIP_OBJ_DIR)/*
	rm -f $(DPC_OBJ_DIR)/*