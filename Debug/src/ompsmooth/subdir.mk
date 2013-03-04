################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/ompsmooth/ompsmooth.c \
../src/ompsmooth/smooth.c 

OBJS += \
./src/ompsmooth/ompsmooth.o \
./src/ompsmooth/smooth.o 

C_DEPS += \
./src/ompsmooth/ompsmooth.d \
./src/ompsmooth/smooth.d 


# Each subdirectory must supply rules for building sources it contributes
src/ompsmooth/%.o: ../src/ompsmooth/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


