.PHONY: all clean

PROJECT_ROOT := $(shell pwd)

include makefiles/c.mk

all: CLI
clean: CLI-clean
