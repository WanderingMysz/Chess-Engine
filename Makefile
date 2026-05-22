.PHONY: all clean

PROJECT_ROOT := $(shell pwd)

include makefiles/cli.mk

all: CLI
clean: CLI-clean
refresh: clean all
