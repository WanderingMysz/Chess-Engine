.PHONY: generate-xslt c-xslt cobol-xslt perl-xslt

RECORD_NAME 	= move_record
XSD				= $(PROJECT_ROOT)/pipeline/move-record/move_record_format.xsd
XSLT_STEM		= $(PROJECT_ROOT)/pipeline/move-record/move_record_to_

C_OUTPUT		= $(PROJECT_ROOT)/chess-logic/include/$(RECORD_NAME).h
COBOL_OUTPUT	= $(PROJECT_ROOT)/pipeline/cobol/$(RECORD_NAME).cpy
PERL_OUTPUT		= $(PROJECT_ROOT)/input-processing/$(RECORD_NAME).pm

generate-xslt: c-xslt cobol-xslt perl-xslt

c-xslt: $(XSLT_STEM)c.xsl $(XSD)
	xsltproc -o $(C_OUTPUT) $(XSLT_STEM)c.xsl $(XSD)

cobol-xslt: $(XSLT_STEM)cobol.xsl $(XSD)
	xsltproc -o $(COBOL_OUTPUT) $(XSLT_STEM)cobol.xsl $(XSD)

perl-xslt: $(XSLT_STEM)perl.xsl $(XSD)
	xsltproc -o $(PERL_OUTPUT) $(XSLT_STEM)perl.xsl $(XSD)
