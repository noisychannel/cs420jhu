################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../ompcompress/compress.c \
../ompcompress/serialcompress.c 

OBJS += \
./ompcompress/compress.o \
./ompcompress/serialcompress.o 

C_DEPS += \
./ompcompress/compress.d \
./ompcompress/serialcompress.d 


# Each subdirectory must supply rules for building sources it contributes
ompcompress/%.o: ../ompcompress/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


