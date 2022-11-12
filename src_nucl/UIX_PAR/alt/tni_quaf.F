CCCCC Quaf-Meister
      SUBROUTINE DRQUAAK
      IMPLICIT REAL*8 (A-H,O-Z)
      INCLUDE 'par/DRQUA_AK'
C     FUER URBANA 3N-POTENTIAL
C     VERSION FUER 6 CLUSTER

C     QUAF BERECHNET DIE REDUZIERTEN ORSTRAUMMATRIXELEMENTE
C     UND MULTIPLIZIERT SIE MIT DEN REDUZIERTEN SPINMATRIX-
C     ELEMENTEN

C     27.8.98 ZUORDNUNG FUER NB KORRIGIERT H.M.H.
C     1.9.98  POTENTIAL UND ZUGEHOERIGE PARAMETER WERDEN JETZT VON EXTRA
C             FILE EINGELESEN, DAMIT NNN-WECHSELWIRKUNG EINFACHER ZU
C             HANDHABEN H.M.H.
C     25.8.99 FUER TUCSON-MELBOURNE 'D1'-TERM WIRD JETZT 'LREG'=2 VERWENDET
C             MIT 'RJK' UND 'QJK' IM POTENTIAL FUER OPERATOR 3 UND 6 H.M.H.
C     29.11.99 ES WERDEN UNTEROPERATOREN MIT LPOT=2 ODER 1 ZUR BEHANDLUNG DER
C             KOORDINATEN BENUTZT, DAZU KOORDINATEN DER WW-TEILCHEN IM 
C             AUFRUF VON BEGRI H.M.H.
C     26.4.00 FUER TEXAS-LOS ALAMOS WIRD DAS PION AUF WECPOT(3) UND DAS OMEGA
C             AUF WECPOT(2) GELEGT IN DER FORM T_PI(R_JK)*(Y_O(R_IK)+Y_O(R_IJ))
C     16.5.00 FUER VPVS WERDEN ZUSAETZLICHE OPERATOREN SPIN-SPIN UND TENSOR
C             AUF DIE OPERATOREN 16 UNE 17 GELEGT H.M.H.

C     1. OPERATOR: NORM
C     2. OPERATOR: ZENTRAL
C     3. OPERATOR: SPIN-SPIN ANTI
C     4. OPERATOR: SPIN-SPIN KOM
C     5. OPERATOR: SPIN-TENSOR R_IK ANTI
C     6. OPERATOR: SPIN-TENSOR R_IJ ANTI
C     7. OPERATOR: SPIN-TENSOR R_IK KOM
C     8. OPERATOR: SPIN-TENSOR R_IJ KOM
C     9. OPERATOR: TENSOR-TENSOR RANG=0 ANTI
C    10. OPERATOR: TENSOR-TENSOR RANG=0 KOM
C    11. OPERATOR: TENSOR-TENSOR RANG=1 ANTI
C    12. OPERATOR: TENSOR-TENSOR RANG=1 KOM
C    13. OPERATOR: TENSOR-TENSOR RANG=2 ANTI
C    14. OPERATOR: TENSOR-TENSOR RANG=2 KOM
C    15. OPERATOR: TENSOR-TENSOR RANG=3 KOM
C    16. OPERATOR: SPIN-SPIN
C    17. OPERATOR: TENSOR

C     INPUT DES PROGRAMMS
C      INTEGERS IM FORMAT 24I3, REALS IM FORMAT 6E12.4

C      BEZEICHNUNG DER BAENDER, AUSDRUCK VARIABLEN
C      FILENAME FUER POTENTIAL,UNIT=25
C     von INPOT=25: DIE STEUERKARTE FUER DIE ZU RECHNENDEN OPERATOREN
C     von INPOT=25: DIE STEUERKARTEN FUER DIE KORRELATIONSFUNKTIONEN
C     von INPOT=25: DIE DEFINITION DER POTENTIALE

C EINGABE FUER DIE FUNKTIONEN

C JEDER INDEX I) ENTSPRICHT EINER KARTE

C DER FOLGENDE DATENSATZ MUSS FUER JEDE FUNKTION (K=1,NZF) WIEDERHOLT
C WERDEN
C  A) ZAHL DER PARAMETERSAETZE                                     NZPAR
C     DIE INNEREN WEITEPARAMETER
C     DIE RADIALWEITEPARAMETER
C     WENN NBAND5=0,WIRD ALLES NEU GERECHNET,WENN NBAND5#0,WIRD VON DIESEM B
C     TEIL UBERNOMMEN

C     NZOPER: ANZAHL DER OPERATOREN IN QUAF
C     NZOPOB:   "     "      "      "  OBER
C     NZOPLU:   "     "      "      "  LUDWIG
C     NZTMAX: MAXIMALE ANZAHL DER TEILCHEN
C     NZFMAX:     "      "     "  ZERLEGUNGEN
C     NZCMAX:     "      "     "  CLUSTER
C     MZGMAX:     "      "     "  SPINFUNKTIONEN
C     NZLWMA:     "      "     "  DREHIMPULSSTRUKTUREN
C     NZRHOM:     "      "     "  BASISVEKTOREN PRO ZERLEGUNG
C     NZPARM:     "      "     "  SAETZTE INNERER WEITEN
C     MZPARM:     "      "     "  RADIALPARAMETER
C     NZPOMA:     "      "     "  POLYNOMSTRUKTUREN
C     NZIQMA:     "      "     "  SIGMAFAKTOREN AUS LUDWIG
C     NZPOTM:     "      "     "  POTENTIALE
C     NPDC:       "      "     "  PDC'S AUS OBER

c    dimensionierung nzavma mag zu klein sein, besser 2*nzcmax hmh ??????
      PARAMETER (NZAVMA=2*NZCMAX-1, NZVMAX=NZTMAX-1,
     *           NZMAT=MZGMAX*MZGMAX,
     *           NDIM4=NZPOMA*NZLWMA,
     *           NZLRH=(2*NZCMAX-3)*(NZCMAX-1),
     * lbig = ndim*ndim*2)
      PARAMETER (LIND=(NZFMAX*(NZFMAX+1))/2*NZOPER)
      DIMENSION INDEX(LIND+1) 


CCCCC Daten, die mit dem Slave geteilt werden

      COMMON /INT/
     $     NZT , MKC  ,  NL , MC3   , 
     $     NR  , NC   , NTE , IRHO  , MFL,
     $     JRHO,  MFR , NDD , MD1   , 
     $     MD2 , MD   , ND  , MD3   , NZV,
     $     IK1 , JK1  , LPOT, MC1   , NCC,
     $     NZAV, NZV1 , NAUS
      PARAMETER( lint=4+5+4+5+5+3 )
      DIMENSION iint(lint)
      EQUIVALENCE( iint(1), nzt )
      
C     Integer Arrays
      COMMON /INTA/
     $     LC(NZTMAX,NPDC),NGRV(2,NZCMAX,NZFMAX),
     $     NUM(5,NZRHOM,NZFMAX),NT1(NPDC), NT2(NPDC),
     $     NT3(NPDC), NZPOT(NZOPER), LREG(NZOPER),
     $     NZRHO(NZFMAX)
      PARAMETER( linta = nztmax*npdc  + 2*nzcmax*nzfmax +
     $     5*nzrhom*nzfmax + npdc + npdc + npdc + nzoper + nzoper +
     $     nzfmax )
      DIMENSION iinta(linta)
      EQUIVALENCE( iinta(1), lc )
      
      COMMON /LUPI/ KVK(NDIM1),
     *    MVK(NDIM1,NZIQMA), IZ, JZ,
     *    KZAHL, LUPAUS, IQM, KZHX(0:NZIQMA)
      PARAMETER( llupi = ndim1 + ndim1*nziqma + 1 + 1 +
     $     1 + 1 + 1 + NZIQMA + 1 )
      DIMENSION ilupi(llupi)
      EQUIVALENCE( ilupi(1), kvk )
     
C     Double Arrays
      COMMON /DOPPELA/
     $     VW(NZTMAX,NZCMAX-1),       SVEC(NZTMAX,NZVMAX),
     $     RVEC(NZVMAX,NZTMAX),       WECPOT(NZPOTM,3,NZOPER),
     $     U(MZGMAX,MZGMAX,NPDC),     CPAR(NZAVMA-1,NZPARM,NZFMAX),
     $     QFCL(NZVMAX,NZVMAX,2*NZCMAX-1),
     $     COF(NZPARM,NZRHOM,NZFMAX), RPAR(MZPARM,NZFMAX)
      PARAMETER( ldoppela = nztmax*(nzcmax-1) + nztmax*nzvmax +
     $     nzvmax*nztmax + nzpotm*3*nzoper + mzgmax*mzgmax*npdc +
     $     (nzavma-1)*nzparm*nzfmax + nzvmax*nzvmax*(2*nzcmax-1) +
     $     nzparm*nzrhom*nzfmax + mzparm*nzfmax )
      DIMENSION rdoppela(ldoppela)
      EQUIVALENCE( rdoppela(1), vw )
      
      COMMON /LUPR/ EPO(NDIM1), WERTL(NDIM4,NDIM4)
      PARAMETER( llupr = ndim1 + ndim4*ndim4 )
      DIMENSION rlupr(llupr)
      EQUIVALENCE( rlupr(1), epo )

CCCCC Ende der mit dem Sklaven gemeinsamen Daten

C     Variablen zur Verwaltung der parallelen Rechnung
      character*255 fnumber, oname
      DIMENSION NRECHNER(1024)
      
C     Ende Variablen zur Verwaltung der parallelen Rechnung

      COMMON /LUR/ NBAND3
C
      COMMON /BIG/ DM(NDIM,NDIM,2)
      dimension rbig(lbig)
      equivalence (rbig(1),dm)
C

      DIMENSION NZC(NZFMAX), MZG(NZFMAX), NOL(NZFMAX),
     $     IENT(NZOPLU), NZPAR(NZFMAX), NREG(NZOPOB),
     $     MS(MZGMAX,NZFMAX), MZPAR(NZFMAX)
               
      DIMENSION KOM(NZOPER,NZFMAX,NZFMAX),
     $     MMASSE(2,MZGMAX,NZFMAX), MLAD(2,MZGMAX,NZFMAX),
     $     MSS(2,MZGMAX,NZFMAX)
      DIMENSION NZLW(NZFMAX), LW(2*NZCMAX-1,NZLWMA,NZFMAX),
     *     LZWERT(5,NZLWMA,NZFMAX), 
     *     NZC1(NZFMAX),  K1VEC(NZFMAX),
     *     NZPO(NZFMAX), KP(NZCMAX-1,NZPOMA,NZFMAX)
      DIMENSION  ITPO(NZFMAX),    JREOB(NZOPER), JRELU(NZOPER)
      DIMENSION kom2(NZFMAX,NZFMAX)

      DIMENSION IMV(NZMAT), UM(NZMAT)

      CHARACTER*80 INFILO, INFILU
      CHARACTER*50 VARFOR(NZOPER)
      CHARACTER*80 CPOTFIL

      DATA VARFOR/'('' NORM'')','('' ZENTRAL '')',
     *     '('' SPIN-SPIN-POTENTIAL ANTI'')',
     *     '('' SPIN-SPIN-POTENTIAL KOM'')',
     *     '('' SPIN-TENSOR-POTENTIAL ANTI R-IK'')',
     *     '('' SPIN-TENSOR-POTENTIAL ANTI R-IJ'')',
     *     '('' SPIN-TENSOR-POTENTIAL KOM R-IK'')',
     *     '('' SPIN-TENSOR-POTENTIAL KOM R-IJ'')',
     *     '('' TENSOR-TENSOR RANG=0 ANTI '')',
     *     '('' TENSOR-TENSOR RANG=0 KOM '')',
     *     '('' TENSOR-TENSOR RANG=1 ANTI '')',
     *     '('' TENSOR-TENSOR RANG=1 KOM '')',
     *     '('' TENSOR-TENSOR RANG=2 ANTI '')',
     *     '('' TENSOR-TENSOR RANG=2 KOM '')',
     *     '('' TENSOR-TENSOR RANG=3 KOM '')',
     *     '('' SPIN-SPIN-POTENTIAL VPVS'')',
     *     '('' SPIN-TENSOR-POTENTIAL VPVS R-IK'')'/

      DATA NZPOT/NZOPER*1/

      DATA JRELU/1,0,0,0,2,0,0,0,3,0,4,0,5,0,6,7,8/
      DATA JREOB/1,0,2,3,4,0,5,6,7,8,9,10,11,12,13,14,15/

      do i=1,1024
         nrechner(i)=0;
      end do
      nhosts = 0
      nteil = 1

      call zeit( cpu_zeit )
      
C     FUER SMIN SIND DIE MIT 'CC' VERSEHENEN KARTEN ZU AKTIVIEREN UND
C     DIE ANGEGEBENEN ZEILEN ZU DEAKTIVIEREN

      OPEN(UNIT=9 ,FILE='DRLUOUT',STATUS='OLD',FORM='UNFORMATTED')
      OPEN(UNIT=10,FILE='DRQUAOUT',STATUS='UNKNOWN',FORM='UNFORMATTED')

      OPEN(UNIT=15 ,FILE='INQUA_N'  ,STATUS='OLD')
      OPEN(UNIT=nout ,FILE='OUTPUT')
      OPEN(UNIT=8 ,FILE='DROBOUT', STATUS='OLD',FORM='UNFORMATTED')

      WRITE(NOUT, 1111)
 1111 FORMAT('1 DRQUA_AK_N VERSION VOM 16.5.00')

      INPUT=15
      INPOT=25
      DO 1   K=1,NZPOTM
         DO 1 L=1,NZOPER
            WECPOT(K,1,L)=1.
            WECPOT(K,2,L)=0.
 1          WECPOT(K,3,L)=0.
      READ (INPUT,1002) NBAND1,NBAND2,NBAND3,IDUM,naufset,NAUS,MOBAUS
     1 ,LUPAUS, ISTOP, NLES

      if(naufset.eq.1) stop 'naufset=1'
      open(unit=18,file='DRFINDEX',status='unknown',form='unformatted')
      if(naufset.eq.0) then
         naufset=1
         nband5=0
         do i=1,lind
           index(i)=-1
         enddo
      else
         nband5=12
            if(nles.eq.0) then
                read(18) nger,(index(i),i=1,nger)
            else
                READ(INPUT,6666) NLES
6666       FORMAT(I6)
                read(18) nger,(index(i),i=1,nles)
           write(nout,*) ' Von ',nger,' gerechneten operatoren werden',
     *          nles,' eingelesen'
                nger=nles
            endif
c     write(nout,*) 'nger',nger,' index ',index(nger)
      do 59 i=1,nger
      if(index(i).lt.naufset) goto 59
      write(nout,*) ' Fuer ',i,' ist index ',index(i),'groesser naufset'
      stop 'naufset falsch'
59    continue
      do i=nger+1,lind
           index(i)=-1
      enddo
      endif

      READ(INPUT,993) CPOTFIL                                                   
 993  FORMAT(A80)                                                               
      OPEN(UNIT=25,FILE=CPOTFIL,STATUS='OLD',FORM='FORMATTED')
      READ(INPOT,1002)(LREG(K),K=1, NZOPER)
 1020 FORMAT(6E12.4)
C     POTENTIAL EINLESEN
      READ (INPOT,1002)  (NZPOT(K), K=2,NZOPER)
      DO 530 K=2,NZOPER
         IF(NZPOT(K).EQ.0) LREG(K) = 0
 530  CONTINUE
      DO 742 MKC=2,NZOPER
         IF(NZPOT(MKC).EQ.0) GOTO 742
         DO 741 K=1,NZPOT(MKC)
 741        READ(INPOT,1020)(WECPOT(K,M,MKC),M=1,3)
 742  CONTINUE
 1042 FORMAT(' ANZAHL DER POTENTIALE',I4)
 1044 FORMAT(//)
 1043 FORMAT(' VORFAKTOR=',G15.6,' BETA IJ ',F11.5,
     1     ' BETA IK  =',F11.5)
C     POTENTIAL EINLESEN
      DO 757 MKC=2,NZOPER
      IF(NZPOT(MKC).EQ.0) GOTO 757
      WRITE(NOUT,1044)
      WRITE(NOUT,VARFOR(MKC))
      WRITE(NOUT,1042) NZPOT(MKC)
      WRITE(NOUT,1043)((WECPOT(I,M,MKC),M=1,3),I=1,NZPOT(MKC))
 757  CONTINUE
      CLOSE(UNIT=25,STATUS='KEEP')
C     EINLESEN DER FUNKTIONSEIGENSCHAFTEN UND PARAMETER
      REWIND  NBAND2
      READ(NBAND2) NZF,NZT,(NREG(K),K=1,NZOPOB),INDEKO
      IF(INDEKO.NE.0) WRITE(NOUT,*)' OBERMATRIXELEMENTE VON EINZELFILES'
      NZV=NZT-1
      IF(MOBAUS.GT.0) WRITE(NOUT,*)' NZF,NZT,(NREG(K),K=1,NZOPOB)',
     *     NZF,NZT,NZV,(NREG(K),K=1,NZOPOB)
         do kr=1,nzfmax
      do kl=1,nzfmax
            kom2(kl,kr)=1
         enddo
      enddo

      IF(NBAND5.LE.0) GOTO 90

      write(nout,*) 'KOM:'
      i=0
      DO 92 KL=1,NZF
      DO 92 KR=1,KL

         KOM2(KL,KR)=0
         DO 91 L = 1, NZOPER
            kom(l,kl,kr)=0
            if(lreg(l).eq.0) goto 91
            i=i+1
            if(index(i).ne.-1) kom(l,kl,kr)=1
            KOM2(KL,KR)=KOM2(KL,KR)+LREG(L)*IABS(1-KOM(L,KL,KR))
91       continue
 92      WRITE(NOUT,*) (KOM(L,KL,KR),L=1,NZOPER),'K2',KOM2(KL,KR)

      write(nout,*) 'war KOM.'

      GO TO 93
 90   CONTINUE
      I=0
      WRITE(NOUT,6100)
      DO 94 KL=1,NZF
         DO 94 KR=1,KL
            DO 94 L=1,NZOPER
      I=I+1
      INDEX(I)=-1
 94   KOM(L,KL,KR)=0
 93   CONTINUE
      NZV1 = NZV - 1
      WECPOT(1,1,1)=2./ FLOAT(NZT*(NZT-1)*(NZT-2))
      DO 20  K = 1,NZF
      READ (NBAND2) NZC(K),MZG(K),NOL(K)
      IF(MOBAUS.GT.0) WRITE(NOUT,*)' NZC(K),MZG(K),NOL(K)',
     *                NZC(K),MZG(K),NOL(K)
      M = MZG(K)
      IF(M.LE.MZGMAX) GOTO 6002
      WRITE(NOUT,*) 'ZERLEGUNG,MZG,MZGMAX',K,MZG(K),MZGMAX
      GO TO 250
 6002 NOL(K)=NOL(K)+1
      READ (NBAND2) ((MMASSE(N,L,K),MLAD(N,L,K),MSS(N,L,K),N=1,2),
     1     MS(L,K),L=1, M)
      IF(MOBAUS.GT.0) WRITE(NOUT,*) 'MMASSE,MLAD,MSS,MS',
     *     ((MMASSE(N,L,K),MLAD(N,L,K),MSS(N,L,K),N=1,2),MS(L,K),L=1, M)
      M = NZC(K)
      READ (NBAND2) ((NGRV(N,L,K),L=1,M),N=1,2)
      IF(MOBAUS.GT.0) WRITE(NOUT,*)'NGRV',((NGRV(N,L,K),L=1,M),N=1,2)
 20   CONTINUE

      REWIND NBAND3
      READ(NBAND3) NZF1,(NZLW(K),NZC1(K),K=1,NZF1),(IENT(K),K=1,NZOPLU),
     1     (NZPO(MH),MH=1,NZF1),INDELU
      IF(INDELU.NE.0) WRITE(NOUT,*)' LUDWMATRIXELEMENTE VON EINZELFILES'
      IF(NAUS.GT.0) WRITE(NOUT,248) NZF1,IENT,(NZLW(K),NZC1(K),NZPO(K),
     1     K=1,NZF1)
 248  FORMAT(' VON LUPO,ZAHL DER ZERLEGUNGEN',I4,' OPERATOREN',7I4,/,
     1     (' ZAHL DER DREHIMPULSE',I4,' ZAHL DER CLUSTER',I4,
     2     'ZAHL DER POLYNOME',I4,/))
      IF(NZF.EQ.NZF1) GOTO 251
      STOP 251
 250  STOP 250
 251  DO 252 K=1,NZF
         ITPO(K)=0
         IF(NZPO(K).NE.0) ITPO(K)=2
         NZPO(K)=MAX0(1,NZPO(K))
         IF(NZC(K).NE.NZC1(K)) STOP 252
 252  CONTINUE

C     UEBERPRUEFUNG, OB GERECHNETE OPERATOREN IN OBER, LUDWIG UND
C     QUAF ZUSAMMENPASSEN

      IF (IENT(1).EQ.0 .OR. NREG(1).EQ.0) STOP 253
      IF (LREG(3).EQ.0) GOTO 255
      IF (NREG(2).EQ.0) STOP 255
 255  IF (LREG(4).EQ.0) GOTO 256
      IF (NREG(3).EQ.0) STOP  256
 256  IF (LREG(5).EQ.0) GOTO 257
      IF (NREG(4)*IENT(2).EQ.0) STOP  257
 257  IF (LREG(6).EQ.0) GOTO 258
      IF (NREG(4)*IENT(2).EQ.0) STOP  258
 258  IF (LREG(7).EQ.0) GOTO 259
      IF (NREG(5)*IENT(2).EQ.0) STOP  259
 259  IF (LREG(8).EQ.0) GOTO 1253
      IF (NREG(6)*IENT(2).EQ.0) STOP 1253
 1253 IF (LREG(9).EQ.0) GOTO 1254
      IF (NREG(7)*IENT(3).EQ.0) STOP 1254
 1254 IF (LREG(10).EQ.0) GOTO 1255
      IF (NREG(8)*IENT(3).EQ.0) STOP 1255
 1255 IF (LREG(11).EQ.0) GOTO 1256
      IF (NREG(9)*IENT(4).EQ.0) STOP 1256
 1256 IF (LREG(12).EQ.0) GOTO 1257
      IF (NREG(10)*IENT(4).EQ.0) STOP 1257
 1257 IF (LREG(13).EQ.0) GOTO 1258
      IF (NREG(11)*IENT(5).EQ.0) STOP 1258
 1258 IF (LREG(14).EQ.0) GOTO 1259
      IF (NREG(12)*IENT(5).EQ.0) STOP 1259
 1259 IF (LREG(15).EQ.0) GOTO 1260
      IF (NREG(13)*IENT(6).EQ.0) STOP 1260
 1260 IF (LREG(16).EQ.0) GOTO 1261
      IF (NREG(14)*IENT(7).EQ.0) STOP 1261
 1261 IF (LREG(17).EQ.0) GOTO 262
      IF (NREG(15)*IENT(8).EQ.0) STOP 262
 262  CONTINUE
C     CHECK OBER/LUDWIG ZU ENDE

C     KONSTRUKTION DER BAHNDREHIMPULSE
      DO 264 K=1,NZF
      M1=2*NZC(K)-3
      M2=NZLW(K)
      IF(M2.LE.NZLWMA) GOTO 6000
      WRITE(NOUT,*) 'ZERLEGUNG,NZLW,NZLWMA',K,NZLW(K),NZLWMA
      GO TO 250
 6000 CONTINUE
      M3=NZC(K)-1
      MPO=NZPO(K)
      READ(NBAND3) ((LW(M,L,K),M=1,M1),L=1,M2),((KP(M,KH,K),M=1,M3),
     1 KH=1,MPO)
      IF(LUPAUS.LT.1) GOTO 263
      DO 1333 L=1,M2
 1333    WRITE (NOUT,1334) (LW(M,L,K),M=1,M1),NOL(K)
 1334 FORMAT(' LWERT, NOL',10I5)
 263  CONTINUE
      DO 265 L=1,M2
      DO 266 M=1,3
 266     LZWERT(M,L,K)=0
      LZWERT(4,L,K)=LW(M3,L,K)
      LZWERT(5,L,K)=LW(M1,L,K)
      GOTO(261,265,267,268,276,282,261), NZC(K)
 261  STOP 'NZC'
C     3-CLUSTER
 267  LZWERT(3,L,K)=LW(1,L,K)
      IF(NOL(K).EQ.NZC(K)) LZWERT(1,L,K)=LW(1,L,K)
      IF(NOL(K).EQ.2) LZWERT(2,L,K)=LW(1,L,K)
      GO TO 265
C     4-CLUSTER
 268  LZWERT(3,L,K)=LW(4,L,K)
      IF(NOL(K).LT.NZC(K)) GOTO 272
      LZWERT(1,L,K)=LZWERT(3,L,K)
      GO TO 265
 272  IF(NOL(K).GT.2) GOTO 275
      LZWERT(2,L,K)=LZWERT(3,L,K)
      GO TO 265
 275  LZWERT(1,L,K)=LW(1,L,K)
      LZWERT(2,L,K)=LW(2,L,K)
      GOTO 265
C     5-CLUSTER
 276  LZWERT(3,L,K)=LW(6,L,K)
      GOTO (277,278,279,280,281,277),NOL(K)
 277  STOP 'NOL'
 278  LZWERT(2,L,K)=LW(6,L,K)
      GOTO 265
 279  LZWERT(1,L,K)=LW(1,L,K)
      LZWERT(2,L,K)=LW(5,L,K)
      GOTO 265
 280  LZWERT(1,L,K)=LW(5,L,K)
      LZWERT(2,L,K)=LW(3,L,K)
      GOTO 265
 281  LZWERT(1,L,K)=LW(6,L,K)
      GOTO 265
C     6-CLUSTER
 282  LZWERT(3,L,K)=LW(8,L,K)
      GOTO (277,283,284,285,286,287,277),NOL(K)
 283  LZWERT(2,L,K)=LW(8,L,K)
      GOTO 265
 284  LZWERT(1,L,K)=LW(1,L,K)
      LZWERT(2,L,K)=LW(7,L,K)
      GOTO 265
 285  LZWERT(1,L,K)=LW(6,L,K)
      LZWERT(2,L,K)=LW(7,L,K)
      GOTO 265
 286  LZWERT(1,L,K)=LW(7,L,K)
      LZWERT(2,L,K)=LW(4,L,K)
      GOTO 265
 287  LZWERT(1,L,K)=LW(8,L,K)
 265  CONTINUE
 264  CONTINUE
C      BAHNDREHIMPULSE ENDE
C      KONSTRUKTION DER BASISVEKTOREN
      I=0
      WRITE (NOUT,1000)
 1000 FORMAT(1H1)
      DO 22  K = 1,NZF
      NZPAR(K) = 0
      MZPAR(K) = 0
       READ(INPUT,1002) NZRHO(K)
      KK=NZRHO(K)
      IF(KK.LE.NZRHOM) GOTO 388
      WRITE(NOUT,*) 'ZERLEGUNG,NZRHO,NZRHOM',K,NZRHO(K),NZRHOM
      GOTO 250
 388  IF(KK.EQ.0) GOTO 22
      M = 2*NZC(K) - 2
      WRITE (NOUT,1001)  K
 1001 FORMAT(//25H WEITEPARAMETER FUER DIE ,  I3,
     1       47H TE CLUSTERSTRUKTUR IN REZIPROKEN FERMIQUADRAT     )
      READ (INPUT,1002)  NZPAR(K) ,MZPAR(K)
 1002 FORMAT(20I3)
      IF(MZPAR(K).GT.MZPARM) STOP 'MZPARM ZU KLEIN'
      LM = NZPAR(K)
      KM=MZPAR(K)
      IF(LM.LE.NZPARM) GOTO 390
      WRITE(NOUT,*) ' ZUVIELE INTERNE PARAMETERSAETZE',NZPAR(K),NZPARM
      GOTO 250
 390  DO 24  L = 1,LM
      READ (INPUT,1003)  (CPAR(N,L,K),N=1,M)
 1003 FORMAT(6E12.4)
 24   WRITE (NOUT,1004)L,(  CPAR(N,L,K),N=1,M)
 1004 FORMAT(/I3,' TER SATZ INNERER WEITEN',/,(8F12.6))
      READ(INPUT,1003)  (RPAR(L,K),L=1,KM)
      DO 125 L=1,KM
      IF(RPAR(L,K).GT.1.E-8) GOTO 125
      WRITE(NOUT,*) ' IN ZERLEGUNG ',K,' IST DIE RELATIVWEITE ',L,
     *              ' FALSCH'
 125  CONTINUE

      WRITE(NOUT,1005)   (RPAR(L,K),L=1,KM)
 1005 FORMAT(/22H SATZ RADIALPARAMETER   /3(8F12.6/))
      DO 240 N=1,KK
      READ(INPUT,1002) NUM(1,N,K),NUM(2,N,K),NUM(4,N,K),NUM(5,N,K)
C     NUM(5,.,.) # 0: BASISVEKTOR WIRD NICHT NEU GERECHNET 

      NUM(4,N,K)=MAX(1,NUM(4,N,K))
      NUM(3,N,K)=N
      READ(INPUT,1003) (COF(L,N,K),L=1,LM)
      WRITE(NOUT,1050) NUM(3,N,K),K,NUM(1,N,K),NUM(2,N,K),NUM(4,N,K),
     1  (COF(L,N,K),L=1,LM)
 1050 FORMAT(//15H DEFINITION DER,I3,22H TEN BASISFUNKTION IST/
     1 I3,13H TE ZERLEGUNG,I3,23H TE SPINISOSPINFUNKTION/I3,
     2 26H TE BAHNDREHIMPULSFUNKTION,I3,' TES POLYNOM',/,
     3 ' SUMMATION UEBER INNERE WEITEN MIT KOEFFIZIENTEN',/,(1P6E12.4))
      IF (NUM(5,N,K).GT.0) WRITE (NOUT,1051)
 240  CONTINUE
      I=I+KK
 22   CONTINUE
C     ENDE DEFINITION BASISVEKTOREN
 2011 FORMAT(////18H BERECHNET WERDEN )
      WRITE (NOUT,2011)
      DO 611 IPV=1,NZOPER
      IF(LREG(IPV).GT.0) WRITE(NOUT,VARFOR(IPV))
 611  CONTINUE
      REWIND NBAND1
      WRITE(NBAND1) NZF,(LREG(K),K=1,NZOPER),I,(NZRHO(K),K=1,NZF)
      DO 950  K = 1,NZF
      M=MZG(K)
      N3=MZPAR(K)
      MC1=NZC(K)-1
      K1VEC(K)=N3
      KK=NZRHO(K)
      IF(KK*N3.LE.NDIM)GOTO 930
      WRITE(NOUT,*) ' NDIM FUER ZERLEGUNG ',NZF,' ZU KLEIN, MINIMAL ',
     *             KK*N3
      STOP 930
 930  IF(KK.LE.0) GOTO 809
      DO 4536 N=1,KK
      N1=NUM(1,N,K)
      N2=NUM(2,N,K)
      N4=NUM(4,N,K)
      WRITE(NBAND1) N3,MMASSE(1,N1,K),MMASSE(2,N1,K),MLAD(1,N1,K),
     $      MLAD(2,N1,K),MSS(1,N1,K),MSS(2,N1,K),MS(N1,K),
     $      (LZWERT(L,N2,K),L=1,5),(RPAR(L,K),L=1,N3),KP(MC1,N4,K)
 4536 CONTINUE
 809     CONTINUE
 950  CONTINUE

      CLOSE( unit=10 )
      if( naus.eq.666 ) stop 'DRQUAOUT geschrieben'
      

         write(fnumber,*) naufset
         do i=1, 255
            if(fnumber(i:i).ne.' ') goto 370
         end do
 370      do j=i, 255
            if(fnumber(j:j).eq.' ') goto 380
         end do
 380      oname = 'DRDMOUT.' // fnumber(i:j)
          open(unit=27,file=oname,status='replace',form='unformatted')

 1933 format(a72)

      IF(INDEKO.NE.0) THEN
         MEFOB=23
         CLOSE(UNIT=8,STATUS='KEEP')
         NBAND2=MEFOB
      ENDIF
      IF(INDELU.NE.0) THEN
         MEFLU=25   
         CLOSE(UNIT=9,STATUS='KEEP')
         NBAND3=MEFLU
      ENDIF
      
C     BESTIMMUNG DER ORTSMATRIXELEMENTE VOR EINSETZEN DER DREHIMPULSE
      icount=0
      DO 40  MFL = 1,NZF
      IZLW=NZLW(MFL)
      IZPW=NZPO(MFL)
      IRHO=NZRHO(MFL)
      IK1=K1VEC(MFL)

      MC = NZC(MFL)
      MC1= MC - 1
      MC2 = MC + MC1
      MC3 = MC2 - 1
      NC= NZPAR(MFL)
      NCC = MZPAR(MFL)

      IF(INDEKO.EQ.0) THEN
         READ(NBAND2)   ((RVEC(M,N),M=1,NZV),N=1,NZT)
         READ(NBAND2)   ((SVEC(N,M),M=1,NZV),N=1,NZT)
         READ(NBAND2)   (((QFCL(M,N,K),M=1,NZV),N=1,NZV),K=1,MC2)
      ENDIF
      
      DO 40 MFR=1,MFL
      CLOSE(UNIT=18,STATUS='KEEP')
      open(unit=18,file='DRFINDEX',status='unknown',form='unformatted')
      REWIND 18
      JZLW=NZLW(MFR)
      JZPW=NZPO(MFR)
      JRHO=NZRHO(MFR)
      JK1=K1VEC(MFR)
      WRITE(NOUT,3009) MFL,MFR
 3009 FORMAT(//,19H ZWISCHEN ZERLEGUNG,I3,15H UND ZERLEGUNG ,I3,
     $24H WIRD ZUR ZEIT GERECHNET )
      MD=NZC(MFR)
      MD1= MD - 1
      
      IF(INDEKO.NE.0) THEN
         READ(INPUT,378) INFILO
 378     FORMAT(A80)
         write(NOUT,*) 'Oeffnen? MFL,MFR,KOM2(MFL,MFR)',
     $        MFL,MFR,KOM2(MFL,MFR),INFILO
         if(KOM2(MFL,MFR).EQ.0) goto 9010
         write(NOUT,*) 'Oeffne ',INFILO
         OPEN(UNIT=MEFOB,FILE=INFILO,STATUS='OLD',FORM='UNFORMATTED')
         REWIND MEFOB
         READ(NBAND2)   ((RVEC(M,N),M=1,NZV),N=1,NZT)
         READ(NBAND2)   ((SVEC(N,M),M=1,NZV),N=1,NZT)
         READ(NBAND2)   (((QFCL(M,N,K),M=1,NZV),N=1,NZV),K=1,MC2)
      ENDIF
9010  IF(INDELU.NE.0) THEN
         READ(INPUT,378) INFILU
        if(KOM2(MFL,MFR).EQ.0) goto 9011

         OPEN(UNIT=MEFLU,FILE=INFILU,STATUS='OLD',FORM='UNFORMATTED')
         REWIND MEFLU
      ENDIF

      if(KOM2(MFL,MFR).EQ.0) goto 9011
      READ(NBAND2)    ((VW(M,K),M=1,NZT),K=1,MD1)
 9011 CONTINUE

      MD2 = MD + MD1
      MD3 = MD2 - 1
      ND = NZPAR(MFR)
      NDD = MZPAR(MFR)
      DO 42 MKC=1, NZOPER
      IZ=(IZPW-1)*NZLWMA+IZLW
      JZ=(JZPW-1)*NZLWMA+JZLW
      NZAV=MC1+MD1
      IF(LREG(MKC).GT.0) WRITE(NOUT,VARFOR(MKC))
      ELREG= LREG(MKC) - 1
C     OPERATOR 1   2    3    4    5    6    7    8    9    10
      GO TO (3008,3008,3008,3008,3004,3004,3004,3004,3002,3002,
     *       3002,3002,3002,3002,3002,3008,3004),MKC 
 3002 NZAV=NZAV+2
      GO TO 3008
 3004 NZAV=NZAV+1
 3008 CONTINUE
      NL = MZG(MFL)
      NR = MZG(MFR)
      IF(JREOB(MKC).EQ.0) GOTO 440
      IF(NREG(JREOB(MKC)).EQ.0) GOTO 440
      IF(KOM2(MFL,MFR).EQ.0) GOTO 440

      READ(NBAND2)    IDUM,NTE,NTE
      IF(NTE.LE.NPDC) GOTO 6029
      WRITE(NOUT,*) ' ZUVIELE PDCS',NTE,' DIMENSION ',NPDC
      STOP 'NPDC'
 6029 CONTINUE
      IF(NTE.EQ.0) GOTO 440
      DO 922   MTE = 1,NTE
      READ(NBAND2) LH,IMQ, (IMV(I),UM(I),I=1,IMQ)
      DO 11 I=1,NL
      DO 11 J=1,NR
      U(I,J,MTE)= 0.
 11   CONTINUE
      DO 12 IK=1,IMQ
      I= IMV(IK)/NR
      J=MOD(IMV(IK),NR)
 12   U(I+1,J+1,MTE)=UM(IK)
      DO 7 I=1,NZT
      LC(NZT+1-I,MTE)=MOD(LH,10)
 7    LH=LH/10 
      NT3(MTE)=MOD(LH,10)
      LH=LH/10 
      NT2(MTE)=MOD(LH,10)
      NT1(MTE)= LH/10 
      IF(MOBAUS.EQ.0) GOTO 922
      IF(LREG(MKC).EQ. 0) GOTO 922
      WRITE (NOUT,838) NT1(MTE), NT2(MTE), NT3(MTE),(LC(K,MTE),K=1,NZT)
 838  FORMAT (10H VON OBER ,20I3)
      IF(MOBAUS.LT.2) GOTO 922
      WRITE (NOUT,839) ((U(NFL,NFR,MTE),NFL=1,NL), NFR=1,NR)
 839  FORMAT (1P10E12.4)
 922  CONTINUE
C
C       READ LUPO ELEMENTE
 440  IF(JRELU(MKC).EQ.0) GOTO 441
      IF(IENT(JRELU(MKC)).EQ.0 ) GOTO 441 
      IF(KOM2(MFL,MFR).EQ.0 .and. INDELU.NE.0) GOTO 441

      CALL LURE(MKC,ITV2)
      IF(LUPAUS.GE.0) WRITE(NOUT,*)' ',ITV2,' LUC-MATRIXELEMENTE'
 441  IF(LREG(MKC).EQ.0)  GOTO 6033
      IF (KOM(MKC,MFL,MFR)) 6030,6031,6032
 6030 WRITE(NOUT,6034)
      GO TO 6033
 6031 WRITE(NOUT,6035)
      GO TO 6033
 6032 WRITE(NOUT,6036)
 6033 CONTINUE
C     Aufsetzen nicht mehr moeglich, einfach entsprechende
C     Operatoren neu rechnen

C     Operator ueberspringen
      IF(KOM(MKC,MFL,MFR).EQ.0) GOTO 95
      if(mfl .ne. mfr) then
         nnn = nzrho(mfl)*nzrho(mfr)
      else
         nnn = (nzrho(mfl)*(nzrho(mfl)+1))/2
      endif
      if(lreg(mkc)*nnn .le. 0) goto 42
      
      IF(KOM(MKC,MFL,MFR).EQ.1) GOTO 942
      WRITE(NOUT,6037)
6037  FORMAT(' FUER DIESEN OPERATOR WIRD ERGAENZT')
         
C     Keine BV
 95   IF(IRHO*JRHO.LE.0) GOTO 42
      IF(LREG(MKC).LE.0) GOTO 42

C     Nichts tun, koennen wir auch selber
      if( nte*itv2.eq.0 ) then
         write( nout,* ) 'Nichts zu tun.',nteil
         index(nteil)=0
         icount=icount+1
               WRITE (27) nteil,icount,index(nteil)


         goto 942
      endif
      

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Andere sollen arbeiten ...

C     Warten, dass sich jemand meldet
      ilen= 1
      call receive_int( idummy,  ilen, isource, 1 )
      idest = isource
      write( nout, * ) 'Empfang von', isource, ': ', idummy
      
c      close(unit=nout)
#ifdef PC_FORTRAN
c     OPEN(UNIT=nout ,FILE='OUTPUT', access='append')
#endif
#ifdef SUN_FORTRAN
c      OPEN(UNIT=nout ,FILE='OUTPUT', access='append')
#endif
#ifdef CRAY_FORTRAN
c      OPEN(UNIT=nout ,FILE='OUTPUT', position='append')
#endif

      nrechner(isource) = nrechner(isource)+1
      if( idummy.eq.9999 ) nhosts=nhosts+1
 
      if(idummy.eq.0) then
          call receive_int(nteilalt, ilen, idest, 0)

          call receive_double(rbig, lbig, idest, 0)
         index(nteilalt)=naufset
         icount=icount+1
               WRITE (27) nteilalt,icount,index(nteilalt)


         write(18) nteil,(index(i),i=1,nteil)
         write(27) (rbig(lb),lb=1,lbig)

      endif
      
C     Dateinamen verschicken
      ilen= 1
      call send_int( nteil, ilen, idest )
      WRITE(NOUT,*) 'Sklave ',idest,' als host ',nhosts,
     *            ' mit operator ',nteil,' gestartet'
         
C     Daten verschicken
      call send_int(    iint    , lint    , idest )
      call send_int(    iinta   , linta   , idest )
      call send_int(    ilupi   , llupi   , idest )
      call send_double( rdoppela, ldoppela, idest )
      call send_double( rlupr   , llupr   , idest )

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC      


 942  nteil = nteil + 1
      
 42   CONTINUE
C     LOOP OPERATOREN
       CLOSE(UNIT=MEFOB,STATUS='KEEP')
 40   CONTINUE
      write(nout,*) 'Alle Arbeit verteilt'
c      close(unit=nout)
#ifdef SUN_FORTRAN
c      OPEN(UNIT=nout ,FILE='OUTPUT', access='append')
#endif
#ifdef CRAY_FORTRAN
c      OPEN(UNIT=nout ,FILE='OUTPUT', position='append')
#endif


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Nachricht zum Beenden der Rechner verschicken
      ilen=1
      null =0
      do ihd = 1, nhosts
         call receive_int( idummy,  ilen, idest, 1 )
          call receive_int(nteilalt, ilen, idest, 0)

          call receive_double(rbig, lbig, idest, 0)  
         index(nteilalt)=naufset
         icount=icount+1
               WRITE (27) nteilalt,icount,index(nteilalt)

         rewind 18
         write(18) nteil,(index(i),i=1,nteil)
         write(27) (rbig(lb),lb=1,lbig)
         write(nout,*) 'Rueckmeldung beenden',nteilalt

         write(nout,*) 'Rechner fertig',idest
         call send_int( null, ilen, idest )
      
c      close(unit=nout)
#ifdef PC_FORTRAN
c     OPEN(UNIT=nout ,FILE='OUTPUT', access='append')
#endif
#ifdef SUN_FORTRAN
c      OPEN(UNIT=nout ,FILE='OUTPUT', access='append')
#endif
#ifdef CRAY_FORTRAN
c      OPEN(UNIT=nout ,FILE='OUTPUT', position='append')
#endif

      end do
      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC     
      rewind 18
         write(18) nteil,(index(i),i=1,nteil)
         write(nout,*) 'nteil',nteil
      close(unit=18,status='keep')
      close(unit=27,status='keep')

      
      WRITE(NOUT,3011)
 3011 FORMAT(//,' ENDE DER RECHNUNG VON QUAF')
      close(unit=nout)
      CLOSE(UNIT=9,STATUS='KEEP')
      CLOSE(UNIT=12,STATUS='KEEP')
      RETURN

 6100 FORMAT(/31H ES WIRD VOELLIG NEU GERECHNET /)
 6034 FORMAT(34H FUER DIESEN OPERATOR WIRD ERSETZT /)
 6035 FORMAT(40H FUER DIESEN OPERATOR WIRD NEU GERECHNET /)
 6036 FORMAT(34H FUER DIESEN OPERATOR WIRD KOPIERT /)
 1051 FORMAT(1X,'DIESER BASISVEKTOR WIRD NICHT NEU GERECHNET')
      END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC      
      SUBROUTINE LURE(MKC,ITV2)
      IMPLICIT REAL*8 (A-H,O-Z)
C     LURE LIEST MATRIXELEMENTE AUS LUDWIG EIN

      INCLUDE 'par/DRQUA_AK'

      PARAMETER (NZAVMA=2*NZCMAX-1, NDIM4=NZPOMA*NZLWMA,
     *           NZLRH=(2*NZCMAX-3)*(NZCMAX-1))

      COMMON /LUPI/ KVK(NDIM1),
     *    MVK(NDIM1,NZIQMA), IZ, JZ, 
     *    KZAHL, LUPAUS, IQM, KZHX(0:NZIQMA)

      COMMON /LUPR/ EPO(NDIM1), WERTL(NDIM4,NDIM4)

      COMMON /LUR/  NBAND3
      
C     Die beiden werden nur von lure, fapor und ordrsrt benoetigt
      COMMON /UPROG/ JQ(NDIM1), INDPO(NDIM1)

      ITV2=0
C     OPERATOR 1  2   3   4   5   6   7   8    9   10
      GOTO (10, 400, 400,400, 10,400,400,400, 10, 400,
     *     10, 400, 10 ,400,10, 10, 10), MKC

 10   DO 30, I=1, IZ
         DO 30, J=1, JZ
            WERTL(I,J)=0.
 30   CONTINUE
      READ (NBAND3) KZAHL, IQMH
      IF (IQMH.GT.NZIQMA) GOTO 445
      IF (LUPAUS.GT.0) WRITE (NOUT,1000)  KZAHL, IQMH
      IF (KZAHL.EQ.0) GOTO 200
      IQM=MAX0(IQMH,1)
      IF (KZAHL.GT.NDIM1) GOTO 444
      READ (NBAND3) (JQ(KW),INDPO(KW),EPO(KW),(MVK(KW,IW),
     *     IW=1,IQM), KW=1,KZAHL)
C    SORTIEREN NACH ANZAHL DER SIGMA FAKTOREN
      CALL ORDSRT(KZAHL)

      CALL FAPOR(1,KZAHL)
 200  CONTINUE

      ITV2=KZAHL

 400  RETURN

 444  WRITE (NOUT,1001)  KZAHL, NDIM1
      STOP 444
 445  WRITE (NOUT,1002)  IQMH, NZIQMA
      STOP 445
 1000 FORMAT(/,' RECORD AUS LUDWIG:',
     *       ' RECORDLAENGE',I7,'   IQM',I5)
 1001 FORMAT(/,1X,'RECORD AUS LUDWIG ZU LANG',/,
     *       ' KZAHL, NDIM1:',3I8)
 1002 FORMAT(/,1X,'ZU VIELE SIGMAFAKTOREN',/,
     *       ' IQM, NZIQMA:',3I4)
      END
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC      
      SUBROUTINE FAPOR(IA,IE)
      IMPLICIT REAL*8 (A-H,O-Z)
C     FAPOR REPRODUZIERT DIE DREIMPULS- UND POLYNOMSTRUKTUR

      INCLUDE 'par/DRQUA_AK'

      PARAMETER (NZAVMA=2*NZCMAX-1, NDIM4=NZPOMA*NZLWMA,
     *           NZLRH=(2*NZCMAX-3)*(NZCMAX-1))

      COMMON /LUPI/ KVK(NDIM1),
     *    MVK(NDIM1,NZIQMA), IZ, JZ, 
     *    KZAHL, LUPAUS, IQM, KZHX(0:NZIQMA)

      COMMON /LUPR/ EPO(NDIM1), WERTL(NDIM4,NDIM4)
      
      COMMON /UPROG/ JQ(NDIM1), INDPO(NDIM1)

      IF(LUPAUS.EQ.0) GOTO 100
      WRITE(NOUT,10) IA,IE
 10   FORMAT(' VON LUDWIG ELEMENTE VON',I8,' BIS ',I10)
      IF(LUPAUS.LT.2) GOTO 100
      DO 15 I=IA,IE
 15      WRITE(NOUT,20) JQ(I),INDPO(I),EPO(I),(MVK(I,J),J=1,JQ(I))
 20   FORMAT(I5,' INDEX',I10,' EPO ',E12.5,' MVK',19I3)
 100  DO 1 I=IA,IE
         KPL=INDPO(I)/100000+1
         KPR=MOD(INDPO(I)/10000,10)+1
         LL=MOD(INDPO(I)/100,100)+1
         LR=MOD(INDPO(I),100)+1
         LPL=(KPL-1)*NZLWMA+LL
         LPR=(KPR-1)*NZLWMA+LR
         WERTL(LPL,LPR) = 1.
         L = NDIM4 * (LPR-1) + LPL
         KVK(I) = L
 1    CONTINUE
      RETURN
      END
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE ORDSRT(KOBEH)
      IMPLICIT REAL*8 (A-H,O-Z)
C     ORDSRT SORTIERT DEN GESAMTEN COMMONBLOCK LUP NACH AUFSTEIGENDER
C     ANZAHL VON SIGMAFAKTOREN
      INCLUDE 'par/DRQUA_AK'

      PARAMETER (NZAVMA=2*NZCMAX-1, NDIM4=NZPOMA*NZLWMA,
     *           NZLRH=(2*NZCMAX-3)*(NZCMAX-1))

      COMMON /LUPI/ KVK(NDIM1),
     *    MVK(NDIM1,NZIQMA), IZ, JZ, 
     *    KZAHL, LUPAUS, IQM, KZHX(0:NZIQMA)

      COMMON /LUPR/ EPO(NDIM1), WERTL(NDIM4,NDIM4)

      COMMON /UPROG/ JQ(NDIM1), INDPO(NDIM1)
      
C    KZHX(IQ) GIBT DIE NIEDRIGSTE ELEMENT NUMMER AN MIT MINDESTENS
C     IQ FAKTOREN IM RECORD

      KUNTH=1
      KZAHL=KOBEH
      IQMH=IQM
      DO 10 IQ=0,IQMH
 10      KZHX(IQ)=0

      DO 100, IQ=0, IQMH
C     LOOP ANZAHL SIGMAFAKTOREN

         KZHX(IQ)=KUNTH
         KUNTEN=KUNTH
         KOBH  =KZAHL

         DO 80, KH=KUNTEN, KZAHL
            IF (JQ(KH).EQ.IQ) GOTO 80
C     SIGMAFAKTOR FALSCH PLAZIERT
            KOBEN=KOBH
            DO 60, IH=KOBEN, KH+1, -1
               IF (JQ(IH).NE.IQ) GOTO 60
C     KANDIDAT ZUM AUSTAUSCH GEFUNDEN
               INT=JQ(IH)
               JQ(IH)=JQ(KH)
               JQ(KH)=INT
               INT=INDPO(IH)
               INDPO(IH)=INDPO(KH)
               INDPO(KH)=INT
               RE=EPO(IH)
               EPO(IH)=EPO(KH)
               EPO(KH)=RE
               DO 20, KQ=1, IQMH
                  IND=MVK(IH,KQ)
                  MVK(IH,KQ)=MVK(KH,KQ)
                  MVK(KH,KQ)=IND
 20            CONTINUE
               KOBH=IH-1
               GOTO 70
               
 60         CONTINUE
C     KEINEN TAUSCHKANDIDATEN MEHR GEFUNDEN, NAECHSTES IQ NEHMEN
            KUNTH=KH
            GOTO 90
            
 70         CONTINUE
C     TAUSCHPARTNER GEFUNDEN UND GETAUSCHT, WEITERSUCHEN
            
 80      CONTINUE
         
C     ENDE LOOP ANZAHL SIGMAFAKTOREN
 90      IF(JQ(KZAHL).EQ.0) GOTO 110
 100  CONTINUE
 110  CONTINUE
      IF(LUPAUS.GT.1) WRITE(NOUT,999) (KZHX(IH),IH=0,IQMH)
 999  FORMAT(' KZHX ',20I4)
      RETURN
      END
