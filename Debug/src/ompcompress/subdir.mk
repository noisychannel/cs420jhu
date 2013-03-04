################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/ompcompress/compress.c \
../src/ompcompress/serialcompress.c 

OBJS += \
./src/ompcompress/compress.o \
./src/ompcompress/serialcompress.o 

C_DEPS += \
./src/ompcompress/compress.d \
./src/ompcompress/serialcompress.d 


# Each subdirectory must supply rules for building sources it contributes
src/ompcompress/%.o: ../src/ompcompress/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


