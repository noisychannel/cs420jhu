################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../build/classes/ompcompress/compress.c \
../build/classes/ompcompress/serialcompress.c 

OBJS += \
./build/classes/ompcompress/compress.o \
./build/classes/ompcompress/serialcompress.o 

C_DEPS += \
./build/classes/ompcompress/compress.d \
./build/classes/ompcompress/serialcompress.d 


# Each subdirectory must supply rules for building sources it contributes
build/classes/ompcompress/%.o: ../build/classes/ompcompress/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


