################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../openmp.examples/block.c \
../openmp.examples/firstprivate.c \
../openmp.examples/howaboutthat.c \
../openmp.examples/loop.c \
../openmp.examples/noscope.c \
../openmp.examples/private.c 

OBJS += \
./openmp.examples/block.o \
./openmp.examples/firstprivate.o \
./openmp.examples/howaboutthat.o \
./openmp.examples/loop.o \
./openmp.examples/noscope.o \
./openmp.examples/private.o 

C_DEPS += \
./openmp.examples/block.d \
./openmp.examples/firstprivate.d \
./openmp.examples/howaboutthat.d \
./openmp.examples/loop.d \
./openmp.examples/noscope.d \
./openmp.examples/private.d 


# Each subdirectory must supply rules for building sources it contributes
openmp.examples/%.o: ../openmp.examples/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


