      PROGRAM DRLUDW
      IMPLICIT double precision (A-H,O-Z)
C     LUDWIG BEREITET DIE BERECHNUNG DER REDUZIERTEN ORTSRAUM-
C     MATRIXELEMENTE VOR
C
C      VERSION FUER 6 CLUSTER MIT DREITEILCHEN KRAEFTEN
C
C     16.5.00 NORM UND TENSOR WERDEN ZUSAETZLICH ALS OPERATOR 7 UND 8
C             KOPIERT H.M.H.
C
C
      INCLUDE 'par/DRLUD'
C
C     DIMENSIONSPARAMETER DER FELDER C, CW, KMV UND
C     DER FELDER DES COMMON-BLOCKS ORD
C     NZFMAX: MAXIMALE ANZAHL DER ZERLEGUNGEN
C     NZCMAX:     "      "     "  CLUSTER
C     NZLWMA:     "      "     "  DREHIMPULSSTRUKTUREN
C     NZIQMA:     "      "     "  SIGMAFAKTOREN
C     LWMAX:  MAXIMALER DREHIMPULSWERT
C     NZOPER: ZAHL DER IM PROGRAMM BERECHNETEN OPERATOREN
C
      PARAMETER (NZLMAX=3*2*(NZCMAX-1))
c     diese dimensionierung mag zu klein sein besser 3*2*nzcmax hmh?????
C     NZLMAX: MAXIMALE ANZAHL DER DREHIMPULSE IM MATR.-EL
C
      COMMON /SDA/ CW(NDIMCW), MVM(NZIQMA,NDIMCW), KAUS,
     *             N5, IQ, NKAPO
C
      COMMON /HIVA/ NBAND
C
      COMMON /MVA/ KV(2*(NZCMAX-1)),KMV(2*(NZCMAX-1),NDIMK),
     *             MMOEG,MKAUS
C
      COMMON /ORD/ EPO(NDIMOR), KZAHL, JQ(NDIMOR),
     *             INDPO(NDIMOR), MVK(NZIQMA,NDIMOR)
C
      COMMON /FAKTOR/ F8(2*LWMAX+1)
C
      COMMON /COMY/ D(100)
C
      DIMENSION IENT(NZOPER)
      DIMENSION NZLW(NZFMAX), NZC(NZFMAX), NZPO(NZFMAX)
      DIMENSION LW(2*NZCMAX-3,NZLWMA,NZFMAX), KRECH(NZFMAX,NZLWMA)
     *          ,KP(NZCMAX-1,NZPOMA,NZFMAX)
      DIMENSION LV(NZLMAX), MV(NZLMAX)
      DIMENSION LKOP1(NZCMAX-2,NZLWMA,NZFMAX),
     *          LKOP2(NZCMAX-2,NZLWMA,NZFMAX)
      DIMENSION C(NDIMC), MW(NDIMC,NZLMAX)
      DIMENSION KWL(2*NZCMAX-3,5), KWR(2*NZCMAX-3,5)
      DIMENSION F3(2*LWMAX+1), F4(4), F5(4), F7(2*LWMAX+1)
C
      CHARACTER*80 INFILE(NZFMAX*NZFMAX)
C
C
      DATA F4/1.,-2.,2.,-2./,F5/1.,.25,.0625,.015625/
C     F4(I+1)=(-)**I (2-DELTA(I,0)),F5(I+1)=1/4**I
C
      OPEN(UNIT=15,FILE='INLU',STATUS='OLD')
      OPEN(UNIT=16,FILE='OUTPUT')
      OPEN(UNIT=9,FILE='DRLUOUT',STATUS='UNKNOWN',FORM='UNFORMATTED')
C      PRINT 1111
C1111  FORMAT('1  DRLUD VERSION VOM 29.2.00')
C
      NKAPO=0
      MMOEG=1
C     NO POLYNOMIALS FOR NKAPO=0
C
C
      D(1)=0.
      D(2)=0.
      DO 100 I=2,99
      HMH=I
100   D(I+1)=LOG(HMH)+D(I)
      INPUT=15
      NBAND=9
      MEFI=23
      FT=SQRT(1.D0/(4.D0*3.141592653589793238D0))
      F7(1)=FT
      F3(1)=1.D0
      F8(1)=1.D0
      FH=1.
      DO 102 NH=2,2*LWMAX+1
      FSH=NH-1
      FH=FSH*FH
      F3(NH)=SQRT(FH)
      F8(NH)=1.D0/FH
      FT=FT*.5D0
102   F7(NH)=FT*SQRT(2.D0*FSH+1.D0)
C     F3(I)=SQRT((I-1)!),F8(I)=1/(I-1)!,F7(I)=(2I-1)/2**(I-1)/SQRT(4PI)
C
C     READ ANGULAR MOMENTUM AND POLYNOMIAL STRUCTURES AND WRITE ON TAPE
1000  FORMAT(20I3)
      READ (INPUT,1000) IDUM,LAUS,KAUS,MAUS,MKAUS,INDEP
C     KAUS =1, ADDITIONAL OUTPUT IN SDA1,LAUS=1,MAUS=1 ADDITIONAL
C     OUTPUT IN MAIN
C        INDEP < 0 HEADER FUER QUAF MIT VIELEN ZERLEGUNGEN SCHREIBEN
C         INDEP > 0 MATRIXELEMENTE FUER VERSCHIEDENE ZERLEGUNGEN AUF
C         FILES, DEREN NAMEN AM ENDE EINGELESEN WERDEN
C         ZUERST DER JOB MIT INDEP >0, DANN DER MIT INDEP <0
C
      IF(INDEP.NE.0) WRITE(NOUT,*) 'INDEP UNGLEICH NULL!  ',INDEP
      READ (INPUT,1000) (IENT(NOP),NOP=1, NZOPER)
C     IENT =1 DETERMINES OPERATORS TO BE CALCULATED,IENT(1..6)= NORM-CENTRAL
C     ,TENSOR,TENSOR**2 RANG=0,TENSOR**2 RANG=1,TENSOR**2 RANG=2,
C      TENSOR**2 RANG=3,kopie norm, KOPIE TENSOR
      IF(IENT(7)+IENT(8).GT.0) OPEN(UNIT=3,STATUS='SCRATCH',
     *                  FORM='UNFORMATTED')
      NBAND3=3
      READ (INPUT,1000) NZF
C     NZF NUMBER OF ZERLEGUNGEN
      DO 110 MH=1,NZF
110   READ (INPUT,1000) NZLW(MH),NZC(MH),NZPO(MH)
C
      REWIND NBAND
      WRITE (NBAND) NZF,(NZLW(MH),NZC(MH),MH=1,NZF)
     * ,(IENT(MH), MH=1, NZOPER), (NZPO(MH), MH=1, NZF),INDEP
C
C     READ AND WRITE ACTUAL ANGULAR MOMENTA AND POLYNOMIALS
C
      WRITE (NOUT,1001) NZF
1001  FORMAT(' ZAHL DER ZERLEGUNGEN',I3)
      IF(NZF.GT.NZFMAX) STOP 1
      DO 140 MH=1,NZF
      WRITE (NOUT,1002) MH
1002  FORMAT(//,'0 ZERLEGUNG',I5)
      N1=NZC(MH)-1
      IF(N1.GT.NZCMAX-1) STOP 2
      N3=N1-1
C     NUMBER OF ANGULAR MOMENTA
      N2=NZLW(MH)
      WRITE (NOUT,1003) NZC(MH),NZLW(MH)
1003  FORMAT(' ANZAHL DER CLUSTER',I5,
     1 ' ANZAHL DER DREHIMPULSSTRUKTUREN',I5)
      IF(NZLW(MH).GT.NZLWMA) STOP 'ZUVIELE DREHIMPULSE'
      DO 130 L=1,N2
      WRITE (NOUT,1004) L
1004  FORMAT(/,' DREHIMPULSSTRUKTUR',I5)
      READ (INPUT,1000)(LW(M,L,MH),M=1,N1),KRECH(MH,L)
C    RELATIV ANGULAR MOMENTA
C
      WRITE (NOUT,1005) (M,LW(M,L,MH),M=1,N1)
1005  FORMAT(I4,' TER DREHIMPULS =',I3,' RELATIVDREHIMPULS')
      IF(N3.LE.0) GOTO 130
      DO 120 M=1,N3
      MM=N1+M
      READ(INPUT,1000) LW(MM,L,MH),LKOP1(M,L,MH),LKOP2(M,L,MH)
C     LW COUPLED ANGULAR MOMENTUM FROM LKOP1 AND LKOP2
      IF(N3.GT.1) GOTO 116
      LKOP1(1,L,MH) =1
      LKOP2(1,L,MH) =2
116   WRITE (NOUT,1006) MM,LW(MM,L,MH),LKOP1(M,L,MH),LKOP2(M,L,MH)
1006  FORMAT(I4,' TER DREHIMPULS =',I3,' GEKOPPELT AUS DREHIMPULS',
     1 I3,' UND DREHIMPULS',I3)
120   CONTINUE
      IF(KRECH(MH,L).GT.0) WRITE(NOUT,2006)
2006  FORMAT(' DIESE DREHIMPULSSTRUKTUR WIRD NICHT NEU GERECHNET')
130   CONTINUE
      MPO=NZPO(MH)
      NZPO(MH)=MAX0(NZPO(MH),1)
      IF(MPO.GT.0)GOTO 124
      DO 122 KH=1,N1
122   KP(KH,1,MH)=0
      MPO=1
      GOTO 127
124   DO 126 KH=1,MPO
126   READ(INPUT,1000) (KP(M,KH,MH),M=1,N1)
      NKAPO=1
127   DO 128 KH=1,MPO
128   WRITE (NOUT,1007) KH,(KP(M,KH,MH),M=1,N1)
1007  FORMAT(' BEI DER',I3,' TEN POLYNOM FUNKTION SIND POLYNOME DER'
     1 ,' ORDNUNG',5I3)
      N4=N1+N3
140   WRITE (NBAND)((LW(M,L,MH),M=1,N4),L=1,N2),
     1 ((KP(M,KH,MH),M=1,N1),KH=1,MPO)
C     END OF INPUT
C     
      IF(INDEP.LT.0) STOP
C     AUSSTIEG FUER KOPF FUER QUAF SCHREIBEN, KEINE BERCHNUNG VON MATRIXELEMENT
C
      IF(INDEP.GT.0) THEN
         ICOUNT= 0   
         DO 277 ILF=1,NZF*NZF
C        IDONE(ILF)= 0
C        IF(NBAND5.NE.0) READ(INPUT,1000) IDONE(ILF)
277      READ(INPUT,278) INFILE(ILF)
278   FORMAT(A48)
         ENDIF
C     START OF CALCULATION OF MATRIX ELEMENTS
      DO 900 MFL=1,NZF
C     LOOP ZERLEGUNGEN LEFT
      N1=NZC(MFL)-1
      N8=N1+N1-1
      N10=N1+1
      N2=NZLW(MFL)
      K1=NZPO(MFL)
      IMFL=MFL
      IF(INDEP.NE.0) IMFL=NZF
      DO 890 MFR=1,IMFL
C     LOOP ZERLEGUNGEN RIGHT
C
      IF(INDEP.NE.0) THEN
         ICOUNT=ICOUNT+1
C        IF(IDONE(ICOUNT).EQ.1) GOTO 890
         NBAND=MEFI
         OPEN(UNIT=MEFI,FILE=INFILE(ICOUNT),STATUS='UNKNOWN',
     *   FORM='UNFORMATTED')
      ENDIF
      N3=NZC(MFR)-1
      N7=N3+1
      N9=N3+N3-1
      N4=NZLW(MFR)
      N5=N1+N3
      NH5=(2*NKAPO+1)*N5
      NH6=NH5+1
      ICOP= 0
      IF(IENT(7)+IENT(8) .GT. 0) ICOP= 1
      IF(ICOP .EQ.1) REWIND NBAND3
      K2=NZPO(MFR)
      DO 880 NOP=1,NZOPER
      IF(ICOP.EQ.1 .AND. NOP.EQ.7) REWIND NBAND3
      IF(ICOP.EQ.1 .AND. NOP.GT.6)  THEN
          READ (NBAND3) KZAHL, IQM
          IF (KZAHL.EQ.0) GOTO 2001
          READ (NBAND3) (JQ(KW),INDPO(KW),EPO(KW),(MVK(IW,KW),
     *                IW=1,IQM), KW=1,KZAHL)
2001      IF(IENT(NOP).EQ.0) GOTO 880
       WRITE (NBAND) KZAHL,IQM
          IF(KZAHL.NE.0) WRITE (NBAND) (JQ(KW),INDPO(KW),EPO(KW),
     *            (MVK(IW,KW), IW=1,IQM), KW=1,KZAHL)
c      DO 4001 KH=1,KZAHL
c      II=JQ(KH)
c      WRITE(NOUT,1037) II,INDPO(KH),EPO(KH),(MVK(IH,KH),IH=1,II)
c1037  FORMAT(I5,' SIGMA-FAKTOREN U INDEX',I8,' EPO=',G15.5,
c     1 'MVK',19I3)
c4001    CONTINUE
          GOTO 880
      ENDIF
      IZREK=0
      KZAHL=0
C     LOOP OF OPERATORS
      IF(IENT(NOP).EQ.0) GOTO 880
      WRITE (NOUT,1008) NOP,MFL,MFR
1008  FORMAT(/,' OPERATOR',I3,' WIRD GERECHNET FUER MFL,MFR',2I3)
C      FOR REDUCING ANGULAR MOMENTA
      DO 800 LL=1,N2
C      LOOP DREHIMPULSSTRUKTUREN LEFT
      NPWL=0
      DO 166 KH=1,N1
166   NPWL=NPWL+LW(KH,LL,MFL)
      NPWL=(-1)**NPWL
      DO 790 LR=1,N4
C     LOOP DREHIMPULSSTRUKTUREN RIGHT
      IF(KRECH(MFL,LL)*KRECH(MFR,LR).GT.0) GOTO 790
C HIER WERDEN SCHON BERECHNETE DREHIMPULSSTRUKTUREN UEBERSPRUNGEN
      NPWR=0
      DO 168 KH=1,N3
168   NPWR=NPWR+LW(KH,LR,MFR)
      NPWR=(-1)**NPWR
C     PARITY CHECK
      IF((NPWL-NPWR).NE.0) GOTO 790
      IDL=LW(N8,LL,MFL)-LW(N9,LR,MFR)
C     IDL TOTAL ANGULAR MOMENTUM DIFFERENCE
      GOTO (170,174,170,172,174,176,171),NOP
171   STOP 4
C     ANGULAR MOMENTUM CHECK
170   IF(IDL)790,176,790
172   IF(IABS(IDL)-1)176,176,790
174   IF(IABS(IDL).GT.2)GOTO 790
C     FACTOR FOR REDUCED MATRIX ELEMENT
176   CONTINUE
      LV(NH5+1)=2
      LV(NH5+2)=2
      IF((NPWL+NPWR)/2+2.GT.NZIQMA) STOP 'NZIQMA ZU KLEIN'
      F=1.
      GOTO (180,186,180,182,186,188,179),NOP
179   STOP 5
180   IX=0
C     NORM,CENTRAL,TENSOR**2 RANG=0
      GOTO 190
182   IX=2
C     TENSOR**2 RANG=1
      GOTO 190
186   IX=4
C     TENSOR,TENSOR**2 RANG=2
      GOTO 190
188   IX=6
C     TENSOR**2 RANG=3
      GOTO 190
190   Y=CLG(2*LW(N9,LR,MFR),IX,2*LW(N8,LL,MFL),2*LW(N9,LR,MFR),2*IDL)
C     MAXIMAL COUPLING
      IF(Y.NE.0.) GOTO 192
      F=0.
      IF(MAUS.LE.0) GOTO 194
      WRITE (NOUT,1010) LW(N9,LR,MFR),IDL,LW(N8,LL,MFL)
1010  FORMAT(' KOPPLUNG LIEFERT NULL',3I5)
      GOTO 790
192   F=F*SQRT(2.*LW(N8,LL,MFL)+1.)/Y
C
C     DETERMINE ANGULAR MOMENTA AND FACTOR
C
194   MV(NH5+1)=IDL
      MIX=IDL
      DO 196 MH=1,N1
      LV(MH)=LW(MH,LL,MFL)
      KH=LV(MH)
196   F=F*F7(KH+1)
C     FACTOR LEFT SIDE
      DO 198 MH=1,N3
      MMH=N1+MH
      LV(MMH)=LW(MH,LR,MFR)
      KH=LV(MMH)
198   F=F*F7(KH+1)
C     FACTOR RIGHT SIDE
      DO 780 KZL=1,K1
C     LOOP POLYNOMIALS LEFT
      DO 199 KHH=1,N1
199   KV(KHH)=KP(KHH,KZL,MFL)
      DO 770 KZR=1,K2
C     LOOP POLYNOMIALS RIGHT
      IF(NKAPO.EQ.0) GOTO 203
      DO 200 KHH=1,N3
      MHH=KHH+N1
200   KV(MHH)=KP(KHH,KZR,MFR)
      DO 201 KHH=1,N5
      IH1=KHH+N5
      IH2=IH1+N5
      LV(IH1)=KV(KHH)
201   LV(IH2)=KV(KHH)
C     HIGH L-VALUES ARE POLYNOMIALS
      CALL MVAL(N5)
203   NP=0
      IF(MAUS.GT.0) WRITE(NOUT,1011) KZL,LL,MFL,NPWL,KZR,LR,MFR,NPWR
1011  FORMAT(/,' MATRIXELEMENT ZWISCHEN ',
     1 'DEM',I3,' TEN POLYNOM DER',I3,
     2 ' TEN DREHIMPULSSTRUKTUR DER',I3,
     3 ' TEN ZERLEGUNG MIT PARITAET',I3,' UND',/,24X,
     4 'DEM',I3,' TEN POLYNOM DER',I3,
     5 ' TEN DREHIMPULSSTRUKTUR DER',I3,
     6 ' TEN ZERLEGUNG MIT PARITAET',I3)
C     SET LEFT M-VALUES TO MINUS L
      DO 222 MH=1,N1
222   MV(MH)=-LV(MH)
      DO 224 MH=N10,N5
224    MV(MH)=LV(MH)
C     DETERMINE ALL POSSIBLE M-VALUES OF LEFT SIDE WITH SUM=TOTAL L
C     AND CALCULATE FACTOR F1
230   F1=1.
      ML=0
      DO 232 MH=1,N1
      KWL(MH,1)=MV(MH)
232   ML=ML+MV(MH)
      IF(ML+LW(N8,LL,MFL).NE.0) GOTO 240
      IF(N1.LE.1) GOTO 254
C     DETERMINE IF COUPLING OF SEVERAL ANGULAR MOMENTA POSSIBLE
      DO 234 MH=N10,N8
      MMH=MH-N1
      MM1=LKOP1(MMH,LL,MFL)
      MM2=LKOP2(MMH,LL,MFL)
      KWL(MH,4)=LW(MM1,LL,MFL)
      KWL(MH,5)=LW(MM2,LL,MFL)
      KWL(MH,2)=KWL(MM1,1)
      KWL(MH,3)=KWL(MM2,1)
       KWL(MH,1)=KWL(MH,2)+KWL(MH,3)
      IF(IABS(KWL(MH,2)+KWL(MH,3)).GT.LW(MH,LL,MFL) ) GOTO 240
234   CONTINUE
      GOTO 250
C     INCREASE MVALUES
240   MH=N1
242   IF(MV(MH)-LV(MH)) 244,246,246
244   MV(MH)=MV(MH)+1
      GOTO 230
246   MV(MH)=-LV(MH)
      MH=MH-1
      IF(MH) 400,400,242
C     ALL COMBINATIONS FOUND
C     CALCULATE FACTOR,D(L,M) AND CLEBSCH
250   ML=MV(1)
      DO 252 MH=2,N1
      M1=LV(MH)+MV(MH)+1
      M2=LV(MH)-MV(MH)+1
      MM=N1+MH-1
      F1=F1*F3(M1)*F3(M2)*CLG(2*KWL(MM,4),2*KWL(MM,5),2*LW(MM,LL,MFL),
     1                       -2*KWL(MM,2),-2*KWL(MM,3))
252   ML=ML+MV(MH)
254   M1=LV(1)+MV(1)+1
      M2=LV(1)-MV(1)+1
      F1=((-1)**ML)*F1*F3(M1)*F3(M2)
C     END LEFT SIDE
C     DETERMINE ALL POSSIBLE M-VALUES OF RIGHT SIDE WITH SUM = TOTAL L
C     AND CALCULATE FACTOR F2
260    F2=1.
      MR=0
      DO 262 MH=N10,N5
      MMH=MH-N10+1
      KWR(MMH,1)=MV(MH)
262   MR=MR+MV(MH)
      IF(MR-LW(N9,LR,MFR).NE.0) GOTO 270
      IF(N3.LE.1) GOTO 284
      DO 264 MH=N7,N9
      MMH=MH-N3
      MM1=LKOP1(MMH,LR,MFR)
      MM2=LKOP2(MMH,LR,MFR)
      KWR(MH,4)=LW(MM1,LR,MFR)
      KWR(MH,5)=LW(MM2,LR,MFR)
      KWR(MH,2)=KWR(MM1,1)
      KWR(MH,3)=KWR(MM2,1)
       KWR(MH,1)=KWR(MH,2)+KWR(MH,3)
      IF(IABS(KWR(MH,2)+KWR(MH,3)).GT.LW(MH,LR,MFR)) GOTO 270
264   CONTINUE
      GOTO 280
C     REDUCE M-VALUES
270   MH=N5
272   IF(MV(MH)+LV(MH)) 276,276,274
274   MV(MH)=MV(MH)-1
C     NEXT COMBINATION
      GOTO 260
276   MV(MH)=LV(MH)
      MH=MH-1
      IF(MH-N1) 240,240,272
C     LOOK FOR NEW COMBINATION ON LEFT SIDE
C     CALCULATE FACTOR RIGHT SIDE,D(L,M) AND CLEBSCH
280   DO 282 MH=2,N3
      MMH=MH+N1
      M1=LV(MMH)+MV(MMH)+1
      M2=LV(MMH)-MV(MMH)+1
      NN=N3+MH-1
282   F2=F2*F3(M1)*F3(M2)*CLG(2*KWR(NN,4),2*KWR(NN,5),2*LW(NN,LR,MFR),
     1                        2*KWR(NN,2),2*KWR(NN,3))
284   M1=LV(N10)+MV(N10)+1
      M2=LV(N10)-MV(N10)+1
      F2=F2*F3(M1)*F3(M2)
C     END RIGHT SIDE
C     CHECK DIMENSION OF C AND MW
      IF(NP.LT.NDIMC) GOTO 286
      WRITE (NOUT,1012) NP,NDIMC
1012  FORMAT(' DIMENSION VON C UND MW ZU KLEIN',2I5)
      STOP 6
C     LOOP M-VALUES OF POLYNOMIALS
286   DO 390 IMH=1,MMOEG
      FAPO=1.
      IF(NKAPO.EQ.0) GOTO 290
      DO 288 JH=1,N5
      JH1=JH+N5
      JH2=JH1+N5
      MV(JH1)=KMV(JH,IMH)
      KM=MV(JH1)
      KL=LV(JH1)
      FAPO=FAPO*F4(KM+1)/F8(KL+KM+1)*F5(KL+1)/F8(KL-KM+1)
288   MV(JH2)=-KMV(JH,IMH)
C     BERECHNUNG DER FAKTOREN C UND NOTIEREN VON C UND DEN M-WERTEN
290   FPO2=F2*FAPO
      GOTO (292,296,300,300,300,300,291),NOP
291   STOP 7
292   NP=NP+1
C     NORM         -----------------------------------------------
      DO 294 KH=1,NH5
294   MW(NP,KH)=MV(KH)
      C(NP)=F1*FPO2*F
      GOTO 390
296   NP=NP+1
C     TENSOR         -----------------------------------------------
      DO 298 KH=1,NH6
298   MW(NP,KH)=MV(KH)
      M1=2+IDL+1
      M2=2-IDL+1
      C(NP)=F1*FPO2*F*F7(3)*F3(M1)*F3(M2)
C     FACTOR FOR TENSOR OPERATOR
      GOTO 390
C     TENSOR**2  ----------------------------------------
 300  DO 310 MVV=-2,2
      NV=MIX-MVV
      FC=CLG(4,4,IX,2*MVV,2*NV)
      IF(ABS(FC).LT.1.E-8) GOTO 310
      NP=NP+1
      DO 302 KH=1,NH5
302   MW(NP,KH)=MV(KH)
      MW(NP,NH5+1)=MVV
      MW(NP,NH5+2)=NV
      C(NP)=F1*FPO2*F*F7(3)*F7(3)*F3(3+MVV)*F3(3-MVV)*F3(3+NV)*F3(3-NV)
     *      *FC
310   CONTINUE
390   CONTINUE
      GOTO 270
C     ALL POSSIBLE M-COMBINATIONS FOUND
C     LOOK FOR POSSIBLE REPRESENTATIONS IN SDA1
400   NQ=NH5
      GOTO(403,402,404,404,404,404,401),NOP
401   STOP 10
404   NQ=NH5+2
       GOTO 403
402   NQ=NH5+1
403   IF(NP.EQ.0)  GOTO 770
      CALL SDA1(LV,MW,NP,C,MZAHL,NQ)
      GOTO 412
412   IF(MZAHL.LE.0) GOTO 770
      IF(MZAHL.LE.NDIMCW) GOTO 414
      WRITE (NOUT,1013) MZAHL,NDIMCW
1013  FORMAT(' DIMENSION VON CW UND MVM ZU KLEIN',2I5)
      STOP 11
414   IF(MAUS.LE.0) GOTO 430
      WRITE (NOUT,1014) MZAHL,IQ
1014  FORMAT(' NUMBER OF MATRIX ELEMENTS',I3,' NUMBER OF ',
     1 'SIGMA-FACTORS',I3)
      WRITE(NOUT,1015)
1015  FORMAT(' MATRIX ELEM SIGMA-FACTORS')
      DO 416 MH=1,MZAHL
416   WRITE (NOUT,1016) CW(MH),(MVM(KH,MH),KH=1,IQ)
1016  FORMAT(1X,E12.4,I5,20I3)
430   CALL PORD(LL,LR,MZAHL,KZL,KZR)
      MDIM=NDIMOR
C
C     END LOOP POLYNOMIALS RIGHT
770   CONTINUE
C     END LOOP POLYNOMIALS LEFT
780   CONTINUE
C     END LOOP DREHIMPULSSTRUKTUREN RIGHT
790   CONTINUE
C     END LOOP DREHIMPULSSTRUKTUREN LEFT
800   CONTINUE
C     DETERMIN IQM
      IQM=0
      IF(KZAHL.LE.0) GOTO 803
      DO 802  KH=1,KZAHL
      IF(IQM.GE.JQ(KH)) GOTO 802
      IQM=JQ(KH)
802   CONTINUE
803    IQM=MAX0(IQM,1)
       WRITE (NBAND) KZAHL,IQM
       IF(ICOP.EQ.1 .AND.( NOP.EQ.1 .OR. NOP.EQ.2)) 
     $      WRITE (NBAND3) KZAHL,IQM
      IF(MAUS + KZAHL .GT.0) WRITE (NOUT,1017) KZAHL,IQM,MDIM
      IZREK=IZREK+1
1017  FORMAT(' KZAHL=',I5,' IQM=',I4,' DIMENSION ',I6)
      IF(KZAHL.LE.0) GOTO 830
      IF(LAUS.GT.0) WRITE (NOUT,1018)
1018  FORMAT(/,' AUSDRUCK DER TERME WIE AUF BAND')
                     CALL WRITAP(LAUS,IQM,ICOP,NOP,NBAND3)
C     END LOOP OPERATORS
830   WRITE (NOUT,879) IZREK
879   FORMAT(I10,' RECORDS GESCHRIEBEN')
880   CONTINUE
C     END LOOP ZERLEGUNGEN RIGHT
890   CONTINUE
C     END LOOP ZERLEGUNGEN LEFT
900   CONTINUE
      WRITE (NOUT,1981)
1981  FORMAT(' ENDE DER RECHNUNG VON LUDWIG')
      STOP
      END
      FUNCTION IVZ(N,M,NP,MP)
      IVZ=1
      IF(ABS(N-M).GT.1) GOTO 100
      IF(N.GT.M) GOTO 50
      IF(NP.LT.MP) RETURN
C     BEIDE PAARE < 0
40    IVZ=-1
      RETURN
50    IF(NP.GT.MP) RETURN
      GOTO 40
100   IF(N.GT.M) GOTO 150
      IF(NP.GT.MP) RETURN
C     EIN PAAR <0, EINS > 0
      GOTO 40
150   IF(NP.LT.MP) RETURN
      GOTO 40
      END
      FUNCTION IVZ8(M,NP,N,MP)
C     VORZEICHEN DER BEITRAEGE FUER LS**2 TENSOR 2
      PARAMETER (NOUT=6)
      MDIF=M-NP
      NDIF=N-MP
      IVZ8= 1
      IF(MDIF*NDIF.LT.0) RETURN
      IVZ8= -1
      IF(MDIF*NDIF.GT.0) RETURN
      WRITE(NOUT,*) ' FEHLER IN IVZ8, M, NP, N, MP ',M,NP,N,MP
      STOP 'IVZ8'
      END
      SUBROUTINE MVAL(N5)
      IMPLICIT double precision (A-H,O-Z)
C     THIS SUBROUTINE CALCULATES ALL POSSIBLE MVALUE COMBINATIONS OF
C     POLYNOMIALS
C
      INCLUDE 'par/DRLUD'
C
C
      COMMON /MVA/ KV(2*(NZCMAX-1)), KMV(2*(NZCMAX-1),NDIMK),
     *             MMOEG, MKAUS
C
      DIMENSION IHV(2*(NZCMAX-1))
C
      MMOEG=1
      DO 10 I=1,N5
      IHV(I)=KV(I)
10    MMOEG=MMOEG*(KV(I)+1)
      IF(MMOEG.GT.NDIMK) GOTO 199
      DO 100 I=1,MMOEG
      DO 20 NH=1,N5
20    KMV(NH,I)=IHV(NH)
      MH=N5
25    IF(IHV(MH).LE.0)  GOTO 30
      IHV(MH)=IHV(MH)-1
      GOTO 100
30    IHV(MH)=KV(MH)
      MH=MH-1
      IF(MH) 40,40,25
40     IF(I.LT.MMOEG)GOTO 200
100   CONTINUE
      IF(MKAUS.EQ.0) GOTO 160
      DO 140  MH=1,MMOEG
140   PRINT 150,(KMV(NH,MH),NH=1,N5)
150    FORMAT(' MVALUES',10I4)
160   RETURN
198   FORMAT(' DIMENSION KMV ZU KLEIN')
199   PRINT 198
200   PRINT 201,I,MMOEG,KV,IHV
201   FORMAT(' NACH',I5,' VERSUCHEN VON',I5,' AUFGEGEBEN',
     1 ' KV,IHV= ',12I3)
      STOP 100
      END
      SUBROUTINE SDA1(L,MW,NEL,C,JJ,ND)
      IMPLICIT double precision (A-H,O-Z)
C     THIS SUBROUTINE CALCULATES ALL POSSIBLE REPRESENTATIONS
C     AND DETERMINES WHAT SIGMA FACTORS EXIST
C     FOR EACH REDUCED MATRIXELEMENT
C
C     L CONTAINS NQ L-VALUES
C     MW CONTAINS ALL M-VALUE COMBINATIONS
C     NEL IS THE NUMBER OF MATRIX ELEMENTS C
C     JJ IS DETERMINED IN SDA1 AND GIVES THE NUMBER OF SIGMA COMBINATIONS
C
      INCLUDE 'par/DRLUD'
C
      COMMON /SDA/ CW(NDIMCW), MVM(NZIQMA,NDIMCW), KAUS,
     *           N5, IQ, NKAPO
      COMMON /FAKTOR/ F8(2*LWMAX+1)
C
      DIMENSION L(3*(2*(NZCMAX-1))),MW(NDIMC,3*(2*(NZCMAX-1)))
      DIMENSION IVM(NZIQMA),IVN(NZIQMA),IG(NZIQMA),C(NDIMC)
      DIMENSION JVM(NZIQMA),JVN(NZIQMA),LQ(3*(2*(NZCMAX-1)))
      DIMENSION KY(3*(2*(NZCMAX-1))),KYS(3*(2*(NZCMAX-1)))
      DIMENSION MKOM(2*(NZCMAX-1)+1,2*(NZCMAX-1)+1)
      DIMENSION IWN(NZIQMA),IWM(NZIQMA)
C
C     CHECK PRINTOUT
      IF(KAUS.LE.0) GOTO 20
      WRITE (NOUT,1001) NEL,ND
1001  FORMAT(' ANZAHL DER MATRIXELEMENTE',I4,'ANZAHL DER',
     1 ' DREHIMPULSE',I4)
      WRITE (NOUT,1002) (L(KH),KH=1,ND)
1002  FORMAT(' DREHIMPULSE',19I3)
      DO 10 KH=1,NEL
10    WRITE (NOUT,1003) C(KH),(MW(KH,NH),NH=1,ND)
1003  FORMAT(' MATRIXELEMENT ',E12.4,' M-WERTE',19I3)
C     PREPARATION
20    JJ=0
      NQ=N5+(ND-(2*NKAPO+1)*N5)
      ITEN=0
      IF(NQ.NE.N5) ITEN=1
C      TREATS TENSOR SEPARATELY
      IH=0
      DO 22 MH=1,NQ
      DO 22 NH=MH,NQ
      IH=IH+1
22    MKOM(MH,NH)=IH
C     MKOM GIVES THE COMBINATION OF SIGMA FACTORS
      IH=0
      DO 24  MH=1,ND
      LQ(MH)=L(MH)
24    IH=IH+L(MH)
      IF(MOD(IH,2).NE.0) GOTO 2000
C     CHECK IF SUM L EVEN
      IQ=IH/2
      IF(IH.GT.0) GOTO 28
      C1=0.
      DO 26 NH=1,NEL
26    C1=C1+C(NH)
      CW(1)=C1
      JJ=1
      MVM(1,1)=0
C     NO SIGMA FACTORS
      RETURN
28    DO 32 NH=1,NEL
      IH=0
      DO 30 MH=1,ND
30    IH=IH+MW(NH,MH)
      IF(IH.NE.0) GOTO 2001
C     CHECK IF SUM M-VALUES ZERO
32    CONTINUE
C     LOOK FOR POSSIBLE SIGMA FACTORS
C     DETERMINE INDICES JVN AND JVM
      IH=1
      MD=ND-1
      M=1
34    JVM(IH)=M
      IF(LQ(M)) 2002,54,36
36    LQ(M)=LQ(M)-1
      IF(IH-1) 2003,40,38
38    IF(M.NE.JVM(IH-1)) GOTO 40
      N=JVN(IH-1)
      GOTO 42
40    N=M+1
42    JVN(IH)=N
      IF(LQ(N)) 2002,48,44
44    LQ(N)=LQ(N)-1
      IF(IH-IQ) 46,66,2003
46    IH=IH+1
C     NEXT FACTOR SIGMA
      GOTO 34
C     LOOK FOR OTHER N
48    IF(N-ND) 50,52,2003
50    N=N+1
      GOTO 42
52    LQ(M)=LQ(M)+1
C     RESTORE L-VALUE AND LOOK FOR OTHER M
54    IF(M-MD)  56,58,2003
56    M=M+1
      GOTO 34
58    IF(IH-1)  2003,999,60
60    IH=IH-1
62    M=JVM(IH)
C     ENTRY POINT FOR SEARCH OF OTHER SIGMA COMBINATION
      N=JVN(IH)
      LQ(N)=LQ(N)+1
      GOTO 48
C     JVM AND JVN DETERMINED
C     DETERMINE INDICES IG
66    C1=0.
C     LOOP OVER ALL M-COMBINATIONS
      DO 500 KE=1,NEL
      IF(KAUS.GT.3) WRITE(NOUT,1500) KE
1500   FORMAT(' LOOP UEBER M-WERTE',I5,' TES M-ELEMENT')
      C2=C(KE)
      DO 70 NH=1,ND
      KY(NH)=L(NH)-1+MW(KE,NH)
70    KYS(NH)=L(NH)-1-MW(KE,NH)
C     KY AND KYS GIVE THE NUMBER OF ARROWS FROM N AND TO N
C     CHOOSE SIGMA FACTOR
C     PREPARATION
      IH=1
      M=JVM(IH)
      N=JVN(IH)
      GOTO 100
80    IF(IH-IQ)  82,200,2003
82    IH=IH+1
      M=JVM(IH)
      N=JVN(IH)
C     CHECK IF SAME SIGMA FACTOR
      IF(M-JVM(IH-1).NE.0) GOTO 100
      IF(N-JVN(IH-1).NE.0) GOTO 100
      IF(IG(IH-1))  140,160,100
C     TRY FOR IG=+1,ARROW FROM M TO N
100   IF(KY(N).LE.0) GOTO 140
      IF(KYS(M).LE.0) GOTO 140
      IG(IH)=1
      KY(N)=KY(N)-2
      KYS(M)=KYS(M)-2
C     NEXT SIGMA FACTOR
      GOTO 80
C     TRY FOR IG=-1,ARROW FROM N TO M
140   IF(KYS(N)) 190,162,142
142   IF(KY(M)) 190,164,144
144   IG(IH)=-1
      KY(M)=KY(M)-2
      KYS(N)=KYS(N)-2
C     NEXT SIGMA FACTOR
      GOTO 80
C     TRY FOR IG=0, NO ARROW
160   IF(KYS(N).LT.0) GOTO 190
162   IF(KY(M).LT.0) GOTO 190
164   IF(KY(N).LT.0) GOTO 190
      IF(KYS(M).LT.0) GOTO 190
      IG(IH)=0
      KY(N)=KY(N)-1
      KYS(N)=KYS(N)-1
      KY(M)=KY(M)-1
      KYS(M)=KYS(M)-1
C     NEXT SIGMA FACTOR
      GOTO 80
190   IF(IH-1) 2003,500,192
192   IH=IH-1
194   N=JVN(IH)
      M=JVM(IH)
      IF(IG(IH))  196,197,195
C     RESTORE IG=+1
195   KY(N)=KY(N)+2
      KYS(M)=KYS(M)+2
C     TRY IG=-1
      GOTO 140
C     RESTORE IG=-1
196   KY(M)=KY(M)+2
      KYS(N)=KYS(N)+2
C     TRY IG=0
      GOTO 164
C     RESTORE IG=0
197   KY(M)=KY(M)+1
      KYS(M)=KYS(M)+1
      KY(N)=KY(N)+1
      KYS(N)=KYS(N)+1
C     TRY PRIOR SIGMA FACTOR
      GOTO 190
C     END OF IG
C     UTILYSE EQUALITY OF SIGMA FACTORS
200    DO 250 IK=1,IQ
C     VM FACTOR
      KH=(JVM(IK)-1)/N5
      IVM(IK)=JVM(IK)-KH*N5*NKAPO
C      NO POLYNOMIAL = NKAPO =0,NO REDUCTION NECESSARY
      IF(ITEN*KH.EQ.3) IVM(IK)=IVM(IK)+N5
      KH=MOD(KH,3)
      IWM(IK)=KH+1
C     VN FACTORS
      KH=(JVN(IK)-1)/N5
      IVN(IK)=JVN(IK)-KH*N5*NKAPO
C      NO POLYNOMIAL = NKAPO =0,NO REDUCTION NECESSARY
      IF(ITEN*KH.EQ.3) IVN(IK)=IVN(IK)+N5
      KH=MOD(KH,3)
      IWN(IK)=KH+1
      IF(KAUS.GT.3)  WRITE (NOUT,1200) IK,JVM(IK),JVN(IK)
     1  ,IVM(IK),IVN(IK),IWM(IK),IWN(IK),IG(IK)
1200  FORMAT(' LOOP W-INDICES, IK,JVM,JVN,IVM,IVN,IWM,IWN,IG =',8I5)
C     ORDER VM .LE. VN
      IF(IVM(IK).LE.IVN(IK)) GOTO 230
      KH=IVM(IK)
      IVM(IK)=IVN(IK)
      IVN(IK)=KH
C     ORDER WM .LE. WN
230   IF(IWM(IK).LE.IWN(IK)) GOTO 250
      KH=IWM(IK)
      IWM(IK)=IWN(IK)
      IWN(IK)=KH
250   CONTINUE
C     DETERMINE CONSTANT
      C3=C2
      NF=2
      DO 320 IK=1,IQ
      IS=IK-1
      IF(IG(IK).NE.0) C3=C3*(-.5)
      IF(IS.EQ.0) GOTO 320
C     SKIP FIRST FACTOR
C     CHECK FOR EQUALITY OF ALL INDICES AND CALCULATE FACULTY
      IF(JVN(IK).NE.JVN(IS)) GOTO 310
      IF(JVM(IK).NE.JVM(IS)) GOTO 310
      IF(IG(IK).NE.IG(IS)) GOTO 310
      NF=NF+1
      GOTO 320
310   C3=C3*F8(NF)
      NF=2
320   CONTINUE
      C3=C3*F8(NF)
      C1=C1+C3
C     FACTOR DETERMINED AND SUMMED UP
      IH=IQ
C     RESTORE KY AND KYS VALUES
      IF(KAUS.GT.1) WRITE (NOUT,1320) C1,IH,(IVM(LX),IVN(LX),LX=1,IH)
1320   FORMAT(' NEUES ELEMENT C1=',G15.5,' IH=',I5,' VM VN',20I3)
      GOTO 194
500   CONTINUE
C     CHECK IF FACTOR DIFFERENT FROM ZERO
      IF(ABS(C1).LT.1.E-10) GOTO 610
      JJ=JJ+1
      IF(JJ.GT.NDIMCW) GOTO 2004
C      ORDER INDICES
      CALL SORT2(IQ,IVM,IVN)
      CW(JJ)=C1
      DO 600 IK=1,IQ
      MH=IVM(IK)
      NH=IVN(IK)
600   MVM(IK,JJ)=MKOM(MH,NH)
C     LOOK FOR DIFFERENT SIGMA FACTORS
      IF(KAUS.GT.1) WRITE(NOUT,1610)JJ,CW(JJ),(MVM(IK,JJ),IK=1,IQ)
1610  FORMAT(I5,'TES M-ELEMENT C =',G15.5,' MVM',19I3)
610   IH=IQ
      GOTO 62
999   CONTINUE
      RETURN
2000  WRITE(NOUT,1004)
1004  FORMAT(' +++ SUMME L-WERTE UNGERADE ++')
      STOP 21
2001  WRITE (NOUT,1005) NH,(MW(NH,IH),IH=1,ND)
1005  FORMAT(' ++ SUMME M-WERTE NICHT NULL, NH=',I4,' M-WERTE',19I3)
      STOP 22
2002  WRITE (NOUT,1006) M,LQ(M)
1006  FORMAT(' DER',I5,'TE L-WERT IST NEGATIV',I5)
      STOP 23
2003  WRITE (NOUT,1007) IH,M,N
1007  FORMAT(' INDICES AUSSERHALB BEREICH,IH,M,N',3I5)
      STOP 24
2004  WRITE (NOUT,1008) JJ
1008  FORMAT(' DIMENSIONIERUNG VON CW ZU KLEIN JJ=',2I5)
      STOP 25
      END
      SUBROUTINE PORD(LL,LR,MZAHL,KZL,KZR)
      IMPLICIT double precision (A-H,O-Z)
C     THIS ROUTINE PUTS ALL MATRIXELEMENTS INTO A SCHEME,ACCORDING TO
C     THE OCCURRING SIGMA FACTORS
C
      INCLUDE 'par/DRLUD'
C
      COMMON /SDA/ CW(NDIMCW), MVM(NZIQMA,NDIMCW), KAUS,
     *             N5, IQ, NKAPO
C
      COMMON /ORD/ EPO(NDIMOR), KZAHL, JQ(NDIMOR),
     *             INDPO(NDIMOR), MVK(NZIQMA,NDIMOR)
C
      COMMON /HIVA/ NBAND
C
      IND=(((KZL-1)*10+KZR-1)*100+LL-1)*100+LR-1
      DO 200 MH=1,MZAHL
C     LOOP OVER ALL ELEMENTS DETERMINED IN SDA1
      IF(KZAHL.LE.0) GOTO 100
      DO 40 KH=1,KZAHL
      IF(IQ.NE.JQ(KH))  GOTO 40
C     CHECK IF COINCIDENCE WITH PREVIOUS ELEMENTS
      IF(IND.NE.INDPO(KH)) GOTO 40
C     CHECK IF SAME ANGULAR MOMENTUM STRUCTURE
      IF(IQ.EQ.0)  GOTO 50
      JH=0
      DO 10 IH=1,IQ
10    JH=JH+IABS(MVK(IH,KH)-MVM(IH,MH))
      IF(JH.EQ.0)  GOTO 50
40    CONTINUE
      GOTO 100
50    EPO(KH)=EPO(KH)+CW(MH)
C     MATRIX ELEMENT EXISTS ALREADY, ADD UP
      GOTO 200
100   KZAHL=KZAHL+1
C     NEW MATRIX ELEMENT
      IF(KZAHL.GT.NDIMOR)  GOTO 1001
      EPO(KZAHL)=CW(MH)
      INDPO(KZAHL)=IND
      JQ(KZAHL)=IQ
      IF(IQ.LE.0)  GOTO 190
C     STORE SIGMA INDICES FOR LATER COMPARISON
      DO 120  IH=1,IQ
120   MVK(IH,KZAHL)=MVM(IH,MH)
      GOTO 200
190   MVK(1,KZAHL)=0
200   CONTINUE
      RETURN
1001  WRITE (NOUT,1002) KZAHL,MH,MZAHL
1002  FORMAT(' DIMENSION IN PORD ZU KLEIN,KZAHL=',2I6,'TES ELEMENT',
     1 'VON',I8,/,' FUER DREHIMPULSSTRUKTUR ',I3,' LINKS UND',I3,
     2 ' RECHTS')
      STOP 41
      END
      SUBROUTINE SORT2(IQ,JVM,JVN)
      IMPLICIT double precision (A-H,O-Z)
C     SORT2 ORDER THE INDICES JVM,JVN LEXICOGRAPHICALLY
C
      INCLUDE 'par/DRLUD'
C
      DIMENSION IND(NZIQMA),JVM(NZIQMA),JVN(NZIQMA)
C
      IF(IQ.LE.1) GOTO 200
      IQ1=IQ-1
C     ORDER ACCORDING TO JVM
      CALL SORTA(1,IQ,JVM,IND)
      CALL PUTA(1,IQ,JVN,IND)
C      ORDER JVN
      JMIN=0
      DO 30 I=1,IQ1
      IF(JVM(I+1).NE.JVM(I)) GOTO 25
      IF(JMIN.NE.0) GOTO 22
      JMIN=I
22    IF(I.NE.IQ1) GOTO 30
      JMAX=I+1
      GOTO 27
25    IF(JMIN.LE.0) GOTO 30
      JMAX=I
27    IF(JMAX.EQ.JMIN) GOTO 30
      CALL SORTA(JMIN,JMAX,JVN,IND)
      JMIN=0
30    CONTINUE
200   RETURN
      END
      SUBROUTINE SORTA(IMIN,IMAX,JQ,IND)
      IMPLICIT double precision (A-H,O-Z)
C     SORTA ORDERS ARRAY J AND GIVES THE ORDERING IN IND
      INCLUDE 'par/DRLUD'
C
      DIMENSION JQ(NZIQMA),IND(NZIQMA)
C
      DO 10 I=IMIN,IMAX
10    IND(I)=I
      IF(IMAX.EQ.IMIN) GOTO 200
      IQ1=IMAX-1
      DO 20 I=IMIN,IQ1
      J=I
15    IF(JQ(J+1).GE.JQ(J)) GOTO 20
      JM=JQ(J)
      ISUB=IND(J)
      JQ(J)=JQ(J+1)
      IND(J)=IND(J+1)
      JQ(J+1)=JM
      IND(J+1)=ISUB
      J=J-1
      IF(J.GE.IMIN) GOTO 15
20    CONTINUE
200   RETURN
      END
      SUBROUTINE PUTA(IMIN,IMAX,JQ,IND)
      IMPLICIT double precision (A-H,O-Z)
C     PUTA INTERCHANGES ELEMENTS IN JQ ACCORDING TO IND
C
      INCLUDE 'par/DRLUD'
C
      DIMENSION JQ(NZIQMA),IND(NZIQMA),KH(NZIQMA)
C
      DO 10 I=IMIN,IMAX
10    KH(I)=JQ(I)
      DO 20 I=IMIN,IMAX
      IH=IND(I)
20    JQ(I)=KH(IH)
      RETURN
      END
      SUBROUTINE WRITAP(LAUS,IQM,ICOP,NOP,NBAND3)
      IMPLICIT double precision (A-H,O-Z)
C
      INCLUDE 'par/DRLUD'
C
      COMMON /HIVA/NBAND
C
      COMMON /ORD/ EPO(NDIMOR), KZAHL, JQ(NDIMOR),
     *             INDPO(NDIMOR), MVK(NZIQMA,NDIMOR)
C
      DO 40 KH=1,KZAHL
      IF(LAUS.LE.0) GOTO 40
      II=JQ(KH)
      WRITE(NOUT,1000) II,INDPO(KH),EPO(KH),(MVK(IH,KH),IH=1,II)
1000  FORMAT(I5,' SIGMA-FAKTOREN U INDEX',I8,' EPO=',G15.5,
     1 'MVK',19I3)
40    CONTINUE
      WRITE (NBAND)(JQ(KH),INDPO(KH),EPO(KH),(MVK(IH,KH),IH=1,IQM),
     1 KH=1,KZAHL)
      IF(ICOP.EQ.1 .AND. NOP.EQ.1 .OR. NOP.EQ.2) WRITE (NBAND3)
     1  (JQ(KH),INDPO(KH),EPO(KH),(MVK(IH,KH),IH=1,IQM), KH=1,KZAHL)
      DO 50 IH=1,IQM
      DO 45 KH=1,KZAHL
45    MVK(IH,KH)=0.
50    CONTINUE
      RETURN
      END
      DOUBLE PRECISION FUNCTION CLG(J1,J2,J3,M1,M2)
C
C     CLG BERECHNET DIE CLEBSCH-GORDAN-KOEFFIZIENTEN
C     (J1/2,M1/2;J2/2,M2/2|J3/2,(M1+M2)/2) NACH
C     EDMONDS 'ANGULAR MOMENTUM IN QUANTUM MECHANICS',
C     PRINCETON, 1960 GLEICHUNGEN (3.10.60), (3.7.3)
C     UND TABELLE 2 (1. GLEICHUNG)
C
C     BENUTZT COMMON /COMY/ MIT DEN LOGRITHMEN DER
C     FAKULTAETEN
C
C     M. UNKELBACH 1989
C     LETZTE AENDERUNG: 06.02.89
C
C
      INTEGER JW1, JW2, JW3, MW1, MW2, MW3, JSUM, JSUM1,
     *        JDIF1, JDIF2, JDIF3, JMSUM1, JMSUM2, JMSUM3,
     *        JMDIF1, JMDIF2, JMDIF3, JJM1, JJM2, IMAX, IMIN,
     *        I, J1, J2, J3, M1, M2
C
      DOUBLE PRECISION FAKLN, CLGH
C
      COMMON /COMY/ FAKLN(0:99)
C     FAKLN(I) = LOG(I!)
C
C
C
C
      JW1=J1
      JW2=J2
      JW3=J3
      MW1=M1
      MW2=M2
C
C     CHECK, OB CLG = 0
      CLG=0.
      IF (JW1.LT.IABS(MW1)) RETURN
      IF (JW2.LT.IABS(MW2)) RETURN
      IF (JW3.GT.JW1+JW2.OR.JW3.LT.IABS(JW1-JW2)) RETURN
      MW3=MW1+MW2
      IF (JW3.LT.IABS(MW3)) RETURN
      JMSUM1=JW1+MW1
      JMSUM2=JW2+MW2
      JMSUM3=JW3+MW3
      IF (MOD(JMSUM1,2).EQ.1) RETURN
      IF (MOD(JMSUM2,2).EQ.1) RETURN
      IF (MOD(JMSUM3,2).EQ.1) RETURN
C
C
      JSUM=(JW1+JW2+JW3)/2
      JSUM1=JSUM+1
      JDIF1=JSUM-JW1
      JDIF2=JSUM-JW2
      JDIF3=JSUM-JW3
C
      IF (IABS(MW1)+IABS(MW2).EQ.0) GOTO 100
C
C     NORMALE CLEBSCH-GORDAN-KOEFFIZIENTEN
      JMSUM1=JMSUM1/2
      JMDIF1=JMSUM1-MW1
      JMSUM2=JMSUM2/2
      JMDIF2=JMSUM2-MW2
      JMSUM3=JMSUM3/2
      JMDIF3=JMSUM3-MW3
      JJM1=JDIF1+JMDIF1
      JJM2=JDIF3-JMDIF1
      IMIN=MAX0(0,-JJM2)
      IMAX=MIN0(JMDIF1,JMDIF3)
C
      CLGH=0.
      DO 50, I=IMIN, IMAX
       CLGH=CLGH+DBLE(1-2*MOD(I,2))*
     *     EXP(FAKLN(JMSUM1+I)+FAKLN(JJM1-I)-FAKLN(I)-FAKLN(JMDIF1-I)-
     *         FAKLN(JMDIF3-I)-FAKLN(JJM2+I))
50    CONTINUE
C
      IF (IMIN.GT.IMAX) CLGH=1.
      CLGH=CLGH*EXP((FAKLN(JDIF3)+FAKLN(JMDIF1)+FAKLN(JMDIF2)+
     *             FAKLN(JMDIF3)+FAKLN(JMSUM3)-FAKLN(JSUM1)-
     *             FAKLN(JDIF1)-FAKLN(JDIF2)-FAKLN(JMSUM1)-
     *             FAKLN(JMSUM2)+FAKLN(JW3+1)-FAKLN(JW3))*.5D0)
      CLG=CLGH*DBLE(1-2*MOD(JMDIF1,2))
C
C     ENDE DER BERECHNUNG FUER NORMALE CLEBSCH-GORDAN-KOEFFIZIENTEN
      RETURN
C
C
C
100   CONTINUE
C     PARITAETSCLEBSCH
C
      IF (MOD(JSUM,2).EQ.1) RETURN
C
      CLGH=EXP((FAKLN(JDIF1)+FAKLN(JDIF2)+FAKLN(JDIF3)-FAKLN(JSUM1)+
     *         FAKLN(JW3+1)-FAKLN(JW3))*.5D0+
     *        FAKLN(JSUM/2)-FAKLN(JDIF1/2)-FAKLN(JDIF2/2)-
     *        FAKLN(JDIF3/2))
      CLG=CLGH*DBLE(1-2*MOD((JSUM+JW1-JW2)/2,2))
C
C
C     ENDE DER RECHNUNG FUER PARITAETSCLEBSCH
      RETURN
      END
