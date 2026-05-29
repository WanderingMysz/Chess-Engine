<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xs="http://www.w3.org/2001/XMLSchema">

<!-- ========================= Define Indentations ========================= -->
    <xsl:variable name="root-indent"    select="'       '"/>
    <xsl:variable name="std-indent"     select="'    '"/>
    
    <xsl:output method="text" encoding="UTF-8"/>
    <xsl:include href="warnings.xsl"/>

    <xsl:template match="/xs:schema">
        <xsl:text>      * </xsl:text>
        <xsl:call-template name="warning-header"/>
        <xsl:apply-templates select="xs:element[@name='move']"/>
    </xsl:template>

    <!-- Match the root 'move' element -->
    <xsl:template match="xs:element[@name='move']">
        <xsl:value-of select="$root-indent"/>
        <xsl:text>01 MOVE-RECORD.&#10;</xsl:text>

        <xsl:apply-templates select="//xs:complexType[@name= current()/@type]">
            <!-- with-param is how variables are passed -->
            <xsl:with-param name="level" select="1"/>
        </xsl:apply-templates>

    </xsl:template>

    <!-- Recursively breakdown complexType -->
    <xsl:template match="xs:complexType">
        <xsl:param name="level"/>

        <xsl:apply-templates select="xs:sequence/xs:element">
            <xsl:with-param name="level" select="$level"/>
        </xsl:apply-templates>
    </xsl:template>

<!-- ======================= Basic Element Matching ======================== -->
    <xsl:template match="xs:element">
        <xsl:param name="level"/>

        <xsl:call-template name="apply-indentation">
            <xsl:with-param name="level" select="$level"/>
        </xsl:call-template>

        <!-- If $level < 2, add leading 0 -->
        <xsl:if test="$level &lt; 2">
            <xsl:text>0</xsl:text>
        </xsl:if>
        <xsl:value-of select="$level * 5"/>
        <xsl:text> </xsl:text>
        <xsl:value-of select="translate(@name, 
                                       'abcdefghijklmnopqrstuvwxyz_', 
                                       'ABCDEFGHIJKLMNOPQRSTUVWXYZ-')"/>

        <!-- Record fields -->
        <xsl:variable name="fieldName" select="@type"/>
        <xsl:variable name="simpleType" 
                      select="//xs:simpleType[@name = $fieldName]"/>
        <xsl:variable name="complexType" 
                      select="//xs:complexType[@name = $fieldName]"/>

        <xsl:choose>
            <xsl:when test="$simpleType">
                <xsl:variable name="fieldLength" 
                              select="$simpleType
                                      /xs:restriction
                                      /xs:length
                                      /@value"/>

                <!-- $fieldName PIC X($fieldLength). -->
                <xsl:text> PIC X(</xsl:text>
                <xsl:value-of select="$fieldLength"/>
                <xsl:text>).&#10;</xsl:text>
            </xsl:when>

            <xsl:when test="$complexType">
                <xsl:text>.&#10;</xsl:text>
                <xsl:apply-templates select="$complexType">
                    <!-- Increase recursion level -->
                    <xsl:with-param name="level" select="$level + 1"/>
                </xsl:apply-templates>
            </xsl:when>

        </xsl:choose>

    </xsl:template>

<!-- ======================== Indentation by Level ========================= -->

    <xsl:template name="apply-indentation">
        <xsl:param name="level"/>

        <xsl:if test="$level = 0">
            <xsl:value-of select="$root-indent"/>
        </xsl:if>

        <xsl:if test="$level &gt; 0">
            <xsl:value-of select="$std-indent"/>
            <xsl:call-template name="apply-indentation">
                <xsl:with-param name="level" select="$level - 1"/>
            </xsl:call-template>
        </xsl:if>
    </xsl:template>

</xsl:stylesheet>
