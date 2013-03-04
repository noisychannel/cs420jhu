################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../build/classes/ompsmooth/ompsmooth.c \
../build/classes/ompsmooth/smooth.c 

OBJS += \
./build/classes/ompsmooth/ompsmooth.o \
./build/classes/ompsmooth/smooth.o 

C_DEPS += \
./build/classes/ompsmooth/ompsmooth.d \
./build/classes/ompsmooth/smooth.d 


# Each subdirectory must supply rules for building sources it contributes
build/classes/ompsmooth/%.o: ../build/classes/ompsmooth/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


