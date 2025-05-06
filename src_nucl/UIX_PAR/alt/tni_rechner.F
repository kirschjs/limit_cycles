CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE CALCU
      IMPLICIT REAL*8 (A-H,O-Z)

      INCLUDE 'par/DRQUA_AK'

      PARAMETER (
     $     NZAVMA=2*NZCMAX-1,
     $     NZVMAX=NZTMAX-1,
     $     NZMAT=MZGMAX*MZGMAX,
     $     NDIM4=NZPOMA*NZLWMA,
     $     NZLRH=(2*NZCMAX-3)*(NZCMAX-1))
C      integer*4 time

CCCCC DIESE DATEN MUESSEN VOM MEISTER UEBERTRAGEN WERDEN
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

CCCCC ENDE DER ZU UEBERTRAGENDEN DATEN

C     Im Sklaven lokale Variablen, evtl. muessen
C     einige an begri und haupt uebergeben werden      

      DIMENSION LC1(NZTMAX), 
     $     QFCR(NZVMAX,NZVMAX,2*NZCMAX-1),
     $     QJK(NZVMAX,NZVMAX), RJK(NZVMAX),UR(NZVMAX),
     $     RIK(NZVMAX),  
     $     QFO(NZVMAX,NZVMAX), POR(NZVMAX,NZVMAX),
     $     QIK(NZVMAX,NZVMAX), POL(NZVMAX,NZVMAX),
     $     UU(MZGMAX,MZGMAX),  POLL(NZVMAX,NZVMAX)

      
      PARAMETER (LBIG=NDIM*NDIM*2)

C     An Unterprogramme uebergeben
      COMMON /CSH/ NSH1(NDIM5), NSH2(NDIM5), SH(NDIM5), NSH
      COMMON /BIG/ DM(NDIM,NDIM,2)
      dimension rbig(lbig)
      equivalence (rbig(1),dm)
C
      COMMON /SKLAVE/ QO(NZVMAX,NZVMAX), VR(NZTMAX,NZCMAX-1)
      COMMON VZ(NZVMAX,NZVMAX), WERT(NDIM4,NDIM4), LPARR
      
CCCCC Buchhaltung fuer paralleles Programm
      CHARACTER*255 oname,  fnumber
      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Datenempfang
      ICHEF = 0
      ITYP = 0
C     Das erste Mal anmelden mit '9999'
      IANMELD = 9999

 665  ilen=1
      call send_int( IANMELD, ilen, ICHEF )

      if(IANMELD.eq.0) then
      call send_int( nteil, ilen,ICHEF)
      call send_double(rbig, lbig, ICHEF)
      endif

      ilen = 1
      call receive_int( nteil, ilen, ICHEF, ITYP )

      if( nteil.eq.0 ) goto 666
      
c      write(fnumber,*) nteil
c      do i=1, 255
c         if(fnumber(i:i).ne.' ') goto 710
c      end do
c 710   do j=i, 255
c         if(fnumber(j:j).eq.' ') goto 810
c      end do
c810     oname= 'Outpot.' // fnumber(i:j)
c       print *,'oname ',oname

      call receive_int(    iint    , lint    , ICHEF, ITYP )
      call receive_int(    iinta   , linta   , ICHEF, ITYP )
      call receive_int(    ilupi   , llupi   , ICHEF, ITYP )
      call receive_double( rdoppela, ldoppela, ICHEF, ITYP )
      call receive_double( rlupr   , llupr   , ICHEF, ITYP )
         
      IANMELD = 0

c     OPEN(UNIT=NOUT,FILE=oname,STATUS='UNKNOWN')

      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC      

 1004 FORMAT(/I3,' TER SATZ INNERER WEITEN',/,(8F12.6))
      
      DO 460 N=1,NDIM
         DO 460 M=1,NDIM
            DM(M,N,1)=.0
 460        DM(M,N,2)=.0
      DO 490 M=1,NZT
 490     LC1(M)=0

      DO 44 MTE=1,NTE
      IF(NAUS.LT.1) GOTO 1491
c     WRITE(NOUT,*) MTE,' TES OBERMATRIXELEMENT'

c     CLOSE(UNIT=NOUT,STATUS='KEEP')
#ifdef PC_FORTRAN
c     OPEN(UNIT=NOUT,FILE=oname,ACCESS='APPEND')
#endif
#ifdef SUN_FORTRAN
c     OPEN(UNIT=NOUT,FILE=oname,ACCESS='APPEND')
#endif
      
#ifdef CRAY_FORTRAN
c     OPEN(UNIT=NOUT,FILE=oname,position='APPEND')
#endif

 1491 MM=0
      DO 491 M=1,NZT
 491     MM=MM+IABS(LC(M,MTE)-LC1(M))
      IF(MM.EQ.0) GOTO 492
      DO 494 M=1,NZT
 494     LC1(M)=LC(M,MTE)
      DO 470 M=1,NZT
         MM=LC(M,MTE)
 470     VR(MM,1)=VW(M,MD1)
      TS=.0
      TD=.0
      NA= 1
      DO 471 M=1,NZT
         TS=TS+ ABS(VR(M,1)+SVEC(M,NZV))
 471     TD=TD+ ABS(VR(M,1)-SVEC(M,NZV))
      IF(TS*TD.GT.0.000001) NA=0
C     PERMUTATION UEBER FRAGMENTGRENZEN
      DO 50   M = 1,NZV
         DO 50   N = 1,M
            DO 51   K= 1,MD2
 51            QFCR(M,N,K) = .0
               DO 50 K=1,MD1
 50               VR(M,K) = .0
      DO 290  K = 1,MD
         I1 = NGRV(1,K,MFR)
         I2 = NGRV(2,K,MFR) + I1 - 1
         IF(NGRV(2,K,MFR).LE.1)GOTO 290
         FZZ = 1./ FLOAT(NGRV(2,K,MFR))
         I3=I2-1
         DO 52  L1 = I1,I3
            I4=L1+1
            DO 52  L2 =I4,I2
               L3 = LC(L1,MTE)
               L4 = LC(L2,MTE)
               DO 52  M = 1,NZV
                  DO 52  N = 1,NZV
                     MM=MIN0(M,N)
                     NN=MAX0(N,M)
 52                  QFCR(NN,MM,K) = QFCR(NN,MM,K) +
     $                    (RVEC(M,L3)-RVEC(M,L4))*
     $                    (RVEC(N,L3)-RVEC(N,L4)) *FZZ
 290  CONTINUE
      DO 53  K = 2,MD
         KI = MD1 + K
         DO 100   M = 1,NZT
            MM = LC(M,MTE)
            DO 100  N = 1,NZV
 100           VR(N,K-1) = VR(N,K-1) + VW(M,K-1) *RVEC(N,MM)
               DO 53   M = 1,NZV
                  DO 53   N = 1,NZV
                     MM=MIN0(M,N)
                     NN=MAX0(N,M)
                     QFCR(NN,MM,KI )=QFCR(NN,MM,KI )+VR(M,K-1)*VR(N,K-1)
 53   CONTINUE
 492  CONTINUE
      DO 54  M = 1,NZV
         UR(M) = RVEC(M,NT1(MTE)) - RVEC(M,NT2(MTE))
         RIK(M) = RVEC(M,NT1(MTE)) - RVEC(M,NT3(MTE))
         RJK(M) = RVEC(M,NT2(MTE)) - RVEC(M,NT3(MTE))
         IF(MKC.GT.15) THEN
C     hier ist rik die jacobikoordinate des dritten teilchens
            RIK(M)= RVEC(M,NT1(MTE))+RVEC(M,NT2(MTE))
     $           - 2.*RVEC(M,NT3(MTE))
            RIK(M)= RIK(M)/SQRT(6.)
         ENDIF
         DO 54  N = 1,NZV
            QIK(M,N)=.0
            QJK(M,N)=.0
 54         QFO(M,N)=.0
c     IF(NAUS.GT.2) WRITE(NOUT,*) 'UR ',(UR(M),M=1,NZV)
c     IF(NAUS.GT.2) WRITE(NOUT,*) 'RIK ',(RIK(M),M=1,NZV)
c     IF(NAUS.GT.2) WRITE(NOUT,*) 'RJK ',(RJK(M),M=1,NZV)
      DO 55   M = 1,NZV
         DO 55   N = 1,NZV
            MM=MIN0(M,N)
            NN=MAX0(N,M)
            QFO(NN,MM) = QFO(NN,MM) + UR(M)*UR(N)
            QIK(NN,MM) = QIK(NN,MM) + RIK(M)*RIK(N)
            QJK(NN,MM) = QJK(NN,MM) + RJK(M)*RJK(N)
 55   CONTINUE
      DO 80 IPOT=1, NZPOT(MKC)
      FAK=WECPOT(IPOT,1,MKC)
      DO 81 M=1,NL
         DO 81 N=1,NR
            UU(M,N)=U(M,N,MTE)
 81   CONTINUE
      DO 60   KPAR = 1,NC
      DO 62   M = 1,NZV
         DO 62   N = 1,NZV
 62         POL(M,N) = .0
      DO 64   M = 1,NZV
         IF(MC3.LE.0) GOTO 64
         DO 63   K = 1,MC3
 63         POL(M,M)=POL(M,M) + CPAR(K,KPAR,MFL)*QFCL(M,M,K)
 64   CONTINUE
      DO 70 LPAR=1,ND
C     FUER UNTEROPERATOREN
      LPOT=0
      IF(LREG(MKC).EQ.2)  LPOT=2

      I=0
      DO 101 M=1,IRHO
         KSL= NUM(1,M,MFL)
         KLL= NUM(2,M,MFL)
         KPL=NUM(4,M,MFL)
         JPL=(KPL-1)*NZLWMA+KLL
         NUML= NUM(3,M,MFL)
         TS = COF(KPAR,M,MFL)
         IF (ABS(TS).LT.1.E-10) GOTO 101
         MADL = (M-1) * IK1
         DO 105 N=1,JRHO
            KSR= NUM(1,N,MFR)
            KLR= NUM(2,N,MFR)
            KPR=NUM(4,N,MFR)
            JPR=(KPR-1)*NZLWMA+KLR
            NUMR= NUM(3,N,MFR)
            IF (MFL.EQ.MFR .AND.NUML.LT.NUMR) GOTO 105
            IF(WERTL(JPL,JPR).NE.1.) GOTO 105
            A = TS * UU(KSL,KSR)*COF(LPAR,N,MFR)
            IF(ABS(A).LT.1.E-20) GOTO 105
            MADR = (N-1) * JK1
            IF (I.LT.NDIM5) GOTO 110
c           WRITE (NOUT,111) I
 111        FORMAT ('0NSH ZU KLEIN', I4)
            STOP
 110        I = I + 1
            NSH1(I) = NDIM * (MADR-1) + MADL
            NSH2(I) = NDIM4 * (JPR-1) + JPL
            SH(I) = A
 105     CONTINUE
 101  CONTINUE
      NSH = I
      IF (NSH.EQ.0) GOTO 70
      DO 72   M = 1,NZV
      DO 72   N = 1,NZV
 72      POR(M,N) = .0
      DO 74  M = 1,NZV
         DO 74  N = M,NZV
            IF(MD3.LE.0) GOTO 74
            DO 73  K = 1,MD3
 73            POR(N,M)=POR(N,M)+CPAR(K,LPAR,MFR)*QFCR(N,M,K)
 74   CONTINUE
 774  CONTINUE
C     EINSPRUNG FUER UNTEROPERATOREN
      DO 67 M=1,NZV
         DO 67  N = M,NZV
            IF(LPOT.EQ.0) POLL(N,M)=POL(N,M)+
     1           WECPOT(IPOT,2,MKC)*QFO(N,M)
     2           +WECPOT(IPOT,3,MKC)*QIK(N,M)
            IF(LPOT.EQ.2) POLL(N,M)=POL(N,M)+
     1           WECPOT(IPOT,2,MKC)*QIK(N,M)
     2           +WECPOT(IPOT,3,MKC)*QJK(N,M)
            IF(LPOT.EQ.1) POLL(N,M)=POL(N,M)+
     1           WECPOT(IPOT,2,MKC)*QFO(N,M)
     2           +WECPOT(IPOT,3,MKC)*QJK(N,M)
 67   CONTINUE
      DO 76 LPARR=1,NDD
      DO 77 M=1,NZV
         DO 77 N=M,NZV
 77         QO(N,M)=POLL(N,M)+POR(N,M)+RPAR(LPARR,MFR)*QFCR(N,M,MD2)
      NB=NA
      IF(NA.EQ.0) GOTO 170
      IF((LPOT.EQ.0 .OR. LPOT.EQ.1).AND.ABS(UR(NZV)).GT.1.E-6) GOTO 160
      IF((LPOT.EQ.0 .OR. LPOT.EQ.2).AND.ABS(RIK(NZV)).GT.1.E-6) GOTO 160
      IF((LPOT.EQ.1 .OR. LPOT.EQ.2).AND.ABS(RJK(NZV)).GT.1.E-6) GOTO 160
      GOTO 170
 160  IF(MKC.EQ.1) GOTO 170
      NB=0
 170  CONTINUE
      IF(LPOT.EQ.0) CALL BEGRI(NB,FAK,UR,RIK)
      IF(LPOT.EQ.1) CALL BEGRI(NB,FAK,RJK,RIK)
      IF(LPOT.EQ.2) CALL BEGRI(NB,FAK,RJK,RIK)
 76   CONTINUE
C     LOOP RADIALWEITEN RECHTS
      IF(LPOT.NE.2) GOTO 70
      LPOT=LPOT-1
      GOTO 774
 70   CONTINUE
C     LOOP INNNERE WEITEN RECHTS
 60   CONTINUE
C      LOOP INNERE WEITEN LINKS
 80   CONTINUE
C       LOOP POTENTIAL
 44   CONTINUE
C       LOOP OBER
 900  CONTINUE
 1021 FORMAT(1X,10E12.5)
C      LOOP BASISVECTOR
c     CLOSE( UNIT=NOUT, status='keep' )
      
      GOTO 665
      
 666  RETURN
 1002 FORMAT(20I3)
      END
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE HAUPT
      IMPLICIT REAL*8 (A-H,O-Z)
C     HAUPT FUEHRT HAUPTACHSENTRANSFORMATION DURCH
C     TRANSFORMATION IN VZ, HAUPTACHSENWERTE IN QO

      INCLUDE 'par/DRQUA_AK'
      PARAMETER (NZVMAX=NZTMAX-1)

      COMMON /INT/
     $     NZT , MKC  ,  NL , MC3   ,
     $     NR  , NC   , NTE , IRHO  , MFL,
     $     JRHO,  MFR , NDD , MD1   ,
     $     MD2 , MD   , ND  , MD3   , NZV,
     $     IK1 , JK1  , LPOT, MC1   , NCC,
     $     NZAV, NZV1 , NAUS
      
      COMMON VZ(NZVMAX,NZVMAX)

      COMMON /SKLAVE/ QO(NZVMAX,NZVMAX), VR(NZTMAX,NZCMAX-1)

      DO 1   K = 1,NZV
         VZ(K,K) = 1.
         IF(NZV-K)   1,1,2
 2       A = .5/QO(K,K)
         K1 = K + 1
         DO 7   M = K1,NZV
            VZ(M,K) = .0
            B = A*QO(M,K)
            DO 5   L = 1,K
 5             VZ(M,L) = VZ(M,L) - B*VZ(K,L)
            QO(M,M) = QO(M,M) - B*B*QO(K,K)
            IF(NZV-M)   7,7,6
 6          M1 = M + 1
            DO 4   N = M1,NZV
 4             QO(N,M) = QO(N,M) - A*QO(M,K)*QO(N,K)
 7       CONTINUE
 1    CONTINUE
      RETURN
      END
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC      
      SUBROUTINE BEGRI(NB,FF1,UR,RIK)
      IMPLICIT REAL*8 (A-H,O-Z)
      INCLUDE 'par/DRQUA_AK'
      PARAMETER (NZAVMA=2*NZCMAX-1, NZVMAX=NZTMAX-1,
     *     NDIM4=NZPOMA*NZLWMA, NZLRH=(2*NZCMAX-3)*(NZCMAX-1))
c
      COMMON /BIG/ DM(NDIM,NDIM,2)
c
      COMMON /CSH/ NSH1(NDIM5), NSH2(NDIM5), SH(NDIM5), NSH
c
      COMMON /SKLAVE/ QO(NZVMAX,NZVMAX), VR(NZTMAX,NZCMAX-1)
c
      COMMON /INT/
     $     NZT , MKC  ,  NL , MC3   , 
     $     NR  , NC   , NTE , IRHO  , MFL,
     $     JRHO,  MFR , NDD , MD1   , 
     $     MD2 , MD   , ND  , MD3   , NZV,
     $     IK1 , JK1  , LPOT, MC1   , NCC,
     $     NZAV, NZV1 , NAUS
c
C     Nur rpar wird verwendet!
      COMMON /DOPPELA/
     $     VW(NZTMAX,NZCMAX-1),       SVEC(NZTMAX,NZVMAX),
     $     RVEC(NZVMAX,NZTMAX),       WECPOT(NZPOTM,3,NZOPER),
     $     U(MZGMAX,MZGMAX,NPDC),     CPAR(NZAVMA-1,NZPARM,NZFMAX),
     $     QFCL(NZVMAX,NZVMAX,2*NZCMAX-1),
     $     COF(NZPARM,NZRHOM,NZFMAX), RPAR(MZPARM,NZFMAX)
c      
      COMMON VZ(NZVMAX,NZVMAX), WERT(NDIM4,NDIM4), LPARR
c
      DIMENSION RIK(NZVMAX), UR(NZVMAX)
c
      DIMENSION P(NZAVMA,NZVMAX), BETA(NZVMAX), 
     *     S((NZAVMA*(NZAVMA+1))/2+1),
     *     ZH(NZAVMA),  SS(NZAVMA*(NZAVMA+1)+2)
c
      DIMENSION WERTT(NDIM4*NDIM4), DMM(NDIM*NDIM,2)
c
      EQUIVALENCE (WERT,WERTT), (DM,DMM)
c
      DATA PHI2 /1.77245385090552/
c
      DO 80   M = 1,NZV
         DO 80   N = 1,NZAV
 80         P(N,M) = 0.
      INZ1=(NZAV*(NZAV+1))/2
      DO 81 M=1,INZ1
 81      S(M)=0.
      CALL HAUPT
      FFF=1.
      IF(NZV1.EQ.0) GOTO 71
       DO 1   M = 1,NZV1
      BETA(M)=1./ SQRT(QO(M,M))
 1    FFF=PHI2*BETA(M)*FFF
 71   CONTINUE
      BETA(NZV) = 1.
c     IF(NAUS.GT.2) WRITE(NOUT,*)' beta ',(beta(n),n=1,nzv)

      DO 2 N=1,MC1
      NN=NZV-MC1+N
      DO 2 M=NN,NZV
 2       P(N,M)=VZ(M,NN)*BETA(M)
C     VZ(M,N) TRANSFORMATIONSMATRIX VON S(N) AUF T(M)
C     VZ(M,N)=T(N,M) EQ 7.9
C     P(M,N)=P(M,N) EQ 7.11
      DO 4 N=1,MD1
         NN=MC1+N
         DO 4 M=1,NZV
            DO 6 K=1,M
 6             P(NN,M)=VR(K,N)*VZ(M,K)+P(NN,M)
 4             P(NN,M)=P(NN,M)*BETA(M)
C      P= TRANSFORMATIONS-MATRIX VON RELATIV AUF DIAGONALKOORDINATEN T
C      BEREITS DIVIDIERT DURCH WURZEL BETA
C   OPERATOR 1  2  3  4  5  6  7  8  9  10
      GOTO ( 10,10,10,10,72,7 ,72,7 ,70,70,
     *       70,70,70,70,70,10,72), MKC

 70   DO 88 M=1,NZV
         DO 90 K=1,M
            P(NZAV-1,M)=P(NZAV-1,M)+UR(K)*VZ(M,K)
 90         P(NZAV,M)=P(NZAV,M)+RIK(K)*VZ(M,K)
         P(NZAV-1,M)=P(NZAV-1,M)*BETA(M)
 88      P(NZAV,M)=P(NZAV,M)*BETA(M)
      GOTO 10
 7    DO 8 M=1,NZV
         DO 9 K=1,M
 9          P(NZAV,M)=P(NZAV,M)+UR(K)*VZ(M,K)
 8    P(NZAV,M)=P(NZAV,M)*BETA(M)
      GOTO 10
 72   DO 82 M=1,NZV
         DO 94 K=1,M
 94         P(NZAV,M)=P(NZAV,M)+RIK(K)*VZ(M,K)
 82   P(NZAV,M)=P(NZAV,M)*BETA(M)
 10   CONTINUE
      DO 3  M = 1,NZAV
 3       ZH(M) = P(M,NZV)
      I=0
      DO 12 M=1,NZAV
         DO 12 N=M,NZAV
            I=I+1
            DO 12   K = 1,NZV1
   12 S(I)=S(I)+P(N,K)*P(M,K)
C     S= SIGMA-FAKTOREN
C     SIGMA(N,N') EQ 7.14
 14   DO 170  KPARR = 1,NCC
      BETA(NZV) = 1./ SQRT(QO(NZV,NZV)+RPAR(KPARR,MFL))
      FFW = FFF *PHI2*BETA(NZV)
      FF=FFW**3*FF1
      DO 91   M = 1,NZAV
 91      P(M,NZV) = ZH(M)*BETA(NZV)
c     IF(NAUS.LT.3) GOTO 136
c     DO 132 M=1,NZV
c132     WRITE(NOUT,*) ' P ',(P(N,M),N=1,NZAV)
 136  I = 0
      DO 93   M = 1,NZAV
         DO 93   N = M,NZAV
            I = I + 1
   93 SS(I) = 2.*(S(I) + P(N,NZV)*P(M,NZV))
c     IF(NAUS.GT.2) WRITE(NOUT,*)' SS',(SS(IH),IH=1,(NZAV*(NZAV+1))/2)
c    *              ,' FF',FF
      CALL MAT(FF,SS)
      I0 = NDIM * LPARR + KPARR
      DO 100 M=1,NSH
         I2 = NSH2(M)
         CC= SH(M) * WERTT(I2)
         IF (CC.EQ.0.) GOTO 100
         I1 = NSH1(M) + I0
         IF (NB) 110,110,111
C     DMM(.,.,2) ENTHAELT ALLE BEITRAEGE
C     DMM(.,.,1) ENTHAELT NUR BEITRAEGE FUER NB=0
 110     DMM(I1,1) = DMM(I1,1) + CC
 111     DMM(I1,2) = DMM(I1,2) + CC
c        IF(NAUS.GT.1) WRITE(NOUT,120) M,I1,I2,SH(M),CC,DMM(I1,1),
c    $        DMM(I1,2)
 120     FORMAT(' M,I1,I2,SH,CC,DMM1,2 ',3I5,E14.6,3E20.14)
 100  CONTINUE
 170  CONTINUE
      RETURN
      END
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC      
      SUBROUTINE MAT(H,S)
      IMPLICIT REAL*8 (A-H,O-Z)
      INCLUDE 'par/DRQUA_AK'

      PARAMETER (NZAVMA=2*NZCMAX-1, NZVMAX=NZTMAX-1,
     *           NDIM4=NZPOMA*NZLWMA, NZLRH=(2*NZCMAX-3)*(NZCMAX-1))

      COMMON /LUPI/ KVK(NDIM1),
     *    MVK(NDIM1,NZIQMA), IZ, JZ,
     *    KZAHL, LUPAUS, IQM, KZHX(0:NZIQMA)

      COMMON /LUPR/ EPO(NDIM1), WERTL(NDIM4,NDIM4)

      COMMON /SKLAVE/ QO(NZVMAX,NZVMAX), VR(NZTMAX,NZCMAX-1)

      COMMON VZ(NZVMAX,NZVMAX), WERT(NDIM4,NDIM4)
C
      COMMON /SCH/ WERT2(NDIM4,NDIM4)
C

      DIMENSION S(NZAVMA*(NZAVMA+1)+2)
      DIMENSION WERTT(NDIM4*NDIM4),WERTT2(NDIM4*NDIM4),F(NDIM1)
      EQUIVALENCE (WERT,WERTT)
      EQUIVALENCE (WERT2,WERTT2)

      DO 4   LR = 1,JZ
         DO 4   LL = 1,IZ
            WERT(LL,LR) = .0
 4          WERT2(LL,LR) = .0
c     IF(LUPAUS.GT.3) WRITE(NOUT,*) ' S ',(S(N),N=1,20),' H ',H
      N=1
      IF(KZAHL.EQ.0) GOTO 1
      IF(ABS(H).LT.1.E-15)  GOTO 1
      DO 20 K=1,KZAHL
 20      F(K)=EPO(K)*H
      IF(KZHX(0).GT.KZHX(1)) GOTO 41
      do M=1,IQM
        ISTART=KZHX(M)
        if(M.NE.IQM) then
          IEND=KZHX(M+1)-1
        else
          IEND=KZAHL
        endif
        IS=1
        if(btest(M,0)) then
*voption nopreload(S)
*voption noprefetch(S)
           do K=ISTART,IEND
              F(K)=F(K)*S(MVK(K,IS))
           ENDDO
           IS=IS+1
        ENDIF
        do MM=IS,M,2
*voption nopreload(S)
*voption noprefetch(S)
           do K=ISTART,IEND
              F(K)=F(K)*S(MVK(K,MM))*S(MVK(K,MM+1))
           ENDDO
        ENDDO
      ENDDO
41    IS=1
      IM=1
      if(btest(KZAHL,0)) then
         WERTT(KVK(IS))=WERTT(KVK(IS)) + F(IS)
         IM=KVK(IS)
         IS=IS+1
      ENDIF
      IN=IM
*voption noprefetch(WERTT,WERTT2)
*voption nopreload(WERTT,WERTT2)
      DO 50 K=IS,KZAHL,2
         WERTT(KVK(K)) = WERTT(KVK(K)) + F(K)
         if(IM.lt.KVK(K)) IM=KVK(K)
         WERTT2(KVK(K+1)) = WERTT2(KVK(K+1)) + F(K+1)
         if(IN.lt.KVK(K+1)) IN=KVK(K+1)
c     IF(LUPAUS.LT.3) GOTO 50
c     IF(ABS(F(K)).LT.1.E-15) GOTO 50
c     WRITE(NOUT,100) KVK(K),
c    ,K,KZAHL,WERTT(KVK(K)),EPO(K),F(K)
 50   CONTINUE
      IQ=MAX(IM,IN)
*voption noprefetch(WERTT,WERTT2)
*voption nopreload(WERTT,WERTT2)
      do k=1,IQ
         WERTT(K)=WERTT(K)+WERTT2(K)
      enddo

100   FORMAT(' KVK,K,K2,W,E,F',3I4,E17.10,2E13.6)
    1 CONTINUE
      RETURN
      END
