
% readoem7memmap.m: use memory mapping to directly read file

gpschan = 16; % number of GPS channels (SV chan 0-15)
qzsschan = 4; % number of QZSS channels
qzssofst = 16; % offset for first QZSS channel (SV Chan 16-19)
sbaschn = 4; % number of SBAS channels
sbasofst = 20; % offset for first SBAS channel (SV chan 20-23)
glochan = 14; % number of GLONASS channels
gloofst1 = 0; % offset for first GLONASS channel (SV chan 0-5)
gloofst2 = 24; % offset for first GLONASS channel (SV chan 24-31)
galchan = 14; % number of Galileo channels
galofst = 6; % offset for first Galileo channel (SV chan 6-19)
beichan = 24; % number of BeiDou channels
beiofst1 = 0; % offset for first BeiDou channel (SV chan 0-11)
beiofst2 = 20; % offset for first BeiDou channel (SV chan 20-31)

% Flag for parsing RANGE log, selectable GNSS constellations
parse_rangeb = 1;
parse_sbas = 0; % parse SBAS RANGE log
parse_qzss = 0; % parse QZSS RANGE log
parse_glo = 1; % parse GLONASS RANGE log
parse_gal = 1; % parse Galileo RANGE log
parse_bei = 1; % parse BeiDou RANGE log

parse_bestpos = 1;
parse_ins = 1;
parse_ppp = 1;
parse_psrpos = 1;
parse_allsqmib = 0;
parse_frontenddata = 1;
parse_satvisb = 1;
parse_vel = 1;
disp_msgids = 0;
check_crc = 1;
C1L1P2L2_like_readobs = 0;
if C1L1P2L2_like_readobs
    decimation_interval = 15; %#ok<*UNRCH> % seconds
end

[FileName,PathName] = uigetfile('*.GPS;*.DAT','Choose OEM-7 format GPS file to read');
if ~ischar(FileName)
    Ipp = [];
    tmv = [];
    return
    elsed
    FileRoot = FileName(1:find(FileName=='.')-1);
end

%build index of found sync byte sequences:
m = memmapfile([PathName FileName]);


disp(['Building index to records of ' PathName FileName]);
datln = length(m.data);

% Parse for sync bits for binary header
nx = find(m.data==170);
nx = nx(nx<datln-9);
nxx = find(m.data(nx+1)==68 & m.data(nx+2)== 18);
nxrec = nx(nxx);
nxrec = nxrec(m.data(nxrec+3)==median(m.data(nxrec+3))); % header length should be the same for all records
nxrec = nxrec(1:end-1);

% get header length using median of first 100 votes (should all be the
% same)
% H = double(median(m.data(nxrec(1:100)+3)));
H = double(median(m.data(nxrec+3)));
msgln = double(typecast(reshape([m.data(nxrec+8)'; m.data(nxrec+9)'],[],1),'uint16'));
nxrec = nxrec(nxrec+msgln+H+3 <= datln);
msgln = double(typecast(reshape([m.data(nxrec+8)'; m.data(nxrec+9)'],[],1),'uint16'));

% short header sync bits:
nxs = find(m.data==170);
nxs = nxs(nxs<datln-9);
nxxs = find(m.data(nxs+1)==68 & m.data(nxs+2)== 19);
nxsrec = nxs(nxxs);
nxsrec = nxsrec(m.data(nxsrec+3)==median(m.data(nxsrec+3))); % header length should be the same for all records
nxsrec = nxsrec(1:end-1);
Hs = 12; % short binary header length
msglns = double(typecast(reshape(m.data(nxsrec+3)',[],1),'uint8'));

%% CRC Checking

if check_crc
% check CRC32 values, keep only good records.
% first generate the CRC32 table of 256 values:
CRC32_POLYNOMIAL = uint32(hex2dec('EDB88320'));
CRC32Table = uint32(zeros(1,256));
for kk = 0:255
    crc = uint32(kk);
    for k = 1:8
        if bitand(crc,1)
            crc = bitxor(bitshift(crc,-1),CRC32_POLYNOMIAL);
        else
            crc = bitshift(crc,-1);
        end
    end
    CRC32Table(kk+1) = crc;
end
% then go through parallel CRC processing:
crcmask = uint32(zeros(1,length(nxrec)));
msglnpHp4 = msgln+H+4;
maxmsglnpHp4 = max(msglnpHp4);
if maxmsglnpHp4 > 5000
    maxmsglnpHp4 = 5000;
end
fprintf('Checking CRC32 values in parallel (%d max record length):  ',maxmsglnpHp4);
progstr = '';
nextprog = 0.00;
tic
for k = 1:maxmsglnpHp4
    if k/maxmsglnpHp4 > nextprog
        for bk = 1:length(progstr)+1
            fprintf('\b');
        end
        progstr = sprintf('%d',round((k/maxmsglnpHp4)*100));
        fprintf([progstr '%%']);
        nextprog = nextprog+0.01-1e-15;
    end
    ln = m.data(nxrec(k<=msglnpHp4)+k-1)';
    Temp1 = bitshift(crcmask(k<=msglnpHp4),-8);
    Temp2 = CRC32Table(bitxor(uint32(ln),bitand(crcmask(k<=msglnpHp4),uint32(255)))+1);
    crcmask(k<=msglnpHp4) = bitxor(Temp1,Temp2);
end
fprintf(' done; %d of %d records removed from index.\n',length(find(crcmask~=0)),length(crcmask));
toc
% keep only the records that passed the CRC32 check and update msgln vector:
nxrec = nxrec(crcmask==0);
msgln = double(typecast(reshape([m.data(nxrec+8)'; m.data(nxrec+9)'],[],1),'uint16'));
end

% repeat same for short binary headers:
crcmask = uint32(zeros(1,length(nxsrec)));
msglnpHsp4 = msglns+Hs+4;
maxmsglnpHsp4 = max(msglnpHsp4);
if maxmsglnpHsp4 > 5000
    maxmsglnpHsp4 = 5000;
end
fprintf('Checking CRC32 values in parallel (%d max record length):  ',maxmsglnpHsp4);
progstr = '';
nextprog = 0.00;
tic
for k = 1:maxmsglnpHsp4
    if k/maxmsglnpHsp4 > nextprog
        for bk = 1:length(progstr)+1
            fprintf('\b');
        end
        progstr = sprintf('%d',round((k/maxmsglnpHsp4)*100));
        fprintf([progstr '%%']);
        nextprog = nextprog+0.01-1e-15;
    end
    ln = m.data(nxsrec(k<=msglnpHsp4)+k-1)';
    Temp1 = bitshift(crcmask(k<=msglnpHsp4),-8);
    Temp2 = CRC32Table(bitxor(uint32(ln),bitand(crcmask(k<=msglnpHsp4),uint32(255)))+1);
    crcmask(k<=msglnpHsp4) = bitxor(Temp1,Temp2);
end
fprintf(' done; %d of %d records removed from index.\n',length(find(crcmask~=0)),length(crcmask));
toc
% keep only the records that passed the CRC32 check and update msgln vector:
nxsrec = nxsrec(crcmask==0);
msglns = double(typecast(reshape(m.data(nxsrec+3)',[],1),'uint8'));

%%

if disp_msgids
    msgid = double(typecast(reshape([m.data(nxrec+4)'; m.data(nxrec+5)'],[],1),'uint16'));
    msid = typecast(reshape([m.data(nxrec+16)'; m.data(nxrec+17)'; m.data(nxrec+18)'; m.data(nxrec+19)'],[],1),'uint32');
    sowid = double(msid)./1000;
    unqmsg = unique(msgid);
    msgv = [630 617 43 570 571 638 653 101 41 848 632 633 37];
%     msgstr = {'AGCSTATSB ONTIME 1','ALLSQMDATAB ONNEW','RANGEB ONTIME 1','RAWGPSSUBFRAMEWPB ONNEW','RAWWAASFRAMEWPB ONNEW',...
%         'RXSECSTATUSB ONTIME 1','SYSTEMLEVELSB ONTIME 1','TIMEB ONTIME 1','RAWEPHEMB ONCHANGED','RXINFOB ONNEW','ALLSQMIB ONNEW'};
    msgstr = {'AGCSTATSB','ALLSQMDATAB','RANGEB','RAWGPSSUBFRAMEWPB','RAWWAASFRAMEWPB',...
        'RXSECSTATUSB','SYSTEMLEVELSB','TIMEB','RAWEPHEMB','RXINFOB','ALLSQMIB','ALLSQMQB','VERSIONB'};
    msgcol = [0 0 1; 1 0 0; 0 .5 0; 0 0 0; .7 .6 .1; .8 0 .6; .4 0 .4; .3 .9 0; .7 .3 .3; 0 .9 .3; .3 .3 .7; .1 .6 .7];
    k = figure;
    set(k,'Position',[233 365 933 583],'PaperPosition',[0.5 0.5 10 6.25]);
    set(gca,'Box','on')
    hold on
    msgy = 0;
    legtxt = {''};
    for msgk = 1:length(unqmsg)
        nxm = find(msgid==unqmsg(msgk));
        if ~isempty(nxm)
            msgy = msgy+1;
            plot(sowid(nxm),msgy.*ones(size(sowid(nxm))),'LineStyle','none','Marker','.','MarkerSize',8,'Color',msgcol(mod(msgk-1,size(msgcol,1))+1,:))
            hold on
            nxv = find(msgv==unqmsg(msgk));
            if ~isempty(nxv)
                legtxt{msgy} = msgstr{nxv};
            else
                legtxt{msgy} = ['Msg ID: ' num2str(unqmsg(msgk))];
            end
        end
    end
    set(gca,'XGrid','on','YGrid','on','FontSize',10)
    set(gca,'YLim',0.5+[0 msgy],'YTick',1:msgy,'YTickLabel',legtxt,'YAxisLocation','right','YDir','reverse');
    p = get(gca,'Position');
    set(gca,'Position',[0.03 p(2) p(3)+(p(1)-0.04) p(4)])
    xlabel('time (GPS sec of week)')
    title(FileName,'Interpreter','none')
    disp('zoom in to desired x range then continue (dbcont)')
    keyboard
    set(gca,'XTickLabel',get(gca,'XTick'),'YLim',0.5+[0 msgy])
    disp('Press continue (or enter dbcont)')
    keyboard
    set(gca,'XTickLabel',get(gca,'XTick'),'YLim',0.5+[0 msgy])
end

% % Info for az,el calculations from type 9 msg
% lat = -30.633064;  % Uralla NSW (LM TT&C site GSBAS omni est. pt. position)
% lon = 151.564677;
% h = 1143; % meters
% lat = 55.19237504;  % CDB-A (new)
% lon = -162.70640332;
% h = 49.684; % meters
lat = 38.86210860;  % Zeta
lon = -77.30239372;
h = 97.140; % meters
% lat = 39.377548;  % Woodbine WBN
% lon = -77.079836;
% h = 173; % meters
% lat = 48.145583;  % Brewster BRE
% lon = -119.692259;
% h = 368; % meters
% lat = 39.514084;  % Littleton LTN
% lon = -105.025182;
% h = 1682.828;
% lat = 38+14/60+40.29530/3600;  % Napa APC
% lon = -122-16/60-44.67830/3600;
% h = 11.380; % meters
% lat = 34+24/60+05.50798/3600;  % Santa Paula SZP
% lon = -119-04/60-25.56655/3600;
% h = 198.349; % meters

esq = 0.0066943800229;
Re = 6378136.3; % (m)
hI = 350000; % assumed height of ionosphere (m)
phi = deg2rad(lat);
lam = deg2rad(lon);
bigN = Re./sqrt(1-esq.*(sin(phi).^2));
X = (bigN + h).*cos(phi).*cos(lam);
Y = (bigN + h).*cos(phi).*sin(lam);
Z = ((1-esq).*bigN + h).*sin(phi);

disp(['Using ' num2str(lat) ', ' num2str(lon) ', ' num2str(h) ' as lat, lon (deg), h (m)']);

% % get ALLMEDLLESTIMATES log index, ID: 611, or uint8([99 2])
% nxrx = find(m.data(nxrec+4)==99 & m.data(nxrec+5)==2);
% nx611 = nxrec(nxrx);
% msgln611 = typecast(reshape([m.data(nx611+8)'; m.data(nx611+9)'],[],1),'uint16');
% obs611 = typecast(reshape([m.data(nx611+H)'; m.data(nx611+H+1)';...
%     m.data(nx611+H+2)'; m.data(nx611+H+3)'],[],1),'uint32');
% ms611 = typecast(reshape([m.data(nx611+16)'; m.data(nx611+17)';...
%     m.data(nx611+18)'; m.data(nx611+19)'],[],1),'uint32');

% msglnchk = double(0.*msgln611+4);
% fprintf('\n');
% for k = 1:length(nx611)
%     if ~mod(k,3600)
%         fprintf('.')
%     end
%     nxoff = 16;
%     for kk = 1:obs611(k)
%         nestimates = double(m.data(nx611(k)+H+nxoff));
%         msglnchk(k)=msglnchk(k)+16+12*nestimates;
%         nxoff = nxoff+nestimates*12+16;
%     end
% end
% fprintf('done.\n')
%%
% get RANGE log index, ID: 43, or uint8([43 0])
nxrx = find(m.data(nxrec+4)==43 & m.data(nxrec+5)==0);
if ~isempty(nxrx) && parse_rangeb
    fprintf('RANGE...')
    nx043 = nxrec(nxrx);
%     nx043 = nx043(obs043 > 0); % temporary fix to weed out RANGEB_1 logs with no observations
    nx043 = nx043(m.data(nx043+6)==uint8(0)); % only ANT1
%     nx043 = nx043(m.data(nx043+6)~=uint8(0)); % only ANT2
    obs043 = typecast(reshape([m.data(nx043+H)'; m.data(nx043+H+1)';...
        m.data(nx043+H+2)'; m.data(nx043+H+3)'],[],1),'uint32');
    msgln043 = typecast(reshape([m.data(nx043+8)'; m.data(nx043+9)'],[],1),'uint16');
    idl043 = double(m.data(nx043+12)')./2;
    wk043 = typecast(reshape([m.data(nx043+14)'; m.data(nx043+15)'],[],1),'uint16');
    ms043 = typecast(reshape([m.data(nx043+16)'; m.data(nx043+17)';...
        m.data(nx043+18)'; m.data(nx043+19)'],[],1),'uint32');
    rxstat043 = typecast(reshape([m.data(nx043+20)'; m.data(nx043+21)';...
        m.data(nx043+22)'; m.data(nx043+23)'],[],1),'uint32');
    
    % the following initialization is to try to avoid resizing matrices:
    [maxobs,maxnx] = max(obs043);
    trkstatx = typecast(reshape([m.data(nx043(maxnx)+H+4+44*maxobs-4); m.data(nx043(maxnx)+H+4+44*maxobs-3);...
        m.data(nx043(maxnx)+H+4+44*maxobs-2); m.data(nx043(maxnx)+H+4+44*maxobs-1)],[],1),'uint32');
    chnnumx = bitand(bitshift(trkstatx,-5),31);

    sow_rge = double(ms043)./1000;

    if maxobs > 70
        disp('maxobs is > 70')
        keyboard
    end
    
    % GPS L1
    PRN_rge1=NaN.*zeros(gpschan,length(nx043));
    C1=NaN.*zeros(gpschan,length(nx043));
    C1std=NaN.*zeros(gpschan,length(nx043));
    L1=NaN.*zeros(gpschan,length(nx043));
    L1std=NaN.*zeros(gpschan,length(nx043));
    D1=NaN.*zeros(gpschan,length(nx043));
    CNo1=NaN.*zeros(gpschan,length(nx043));
    locktime1=NaN.*zeros(gpschan,length(nx043));
    parity1=NaN.*zeros(gpschan,length(nx043));
    satsys1=NaN.*zeros(gpschan,length(nx043));
    
    % GPS L2
    PRN_rge2=NaN.*zeros(gpschan,length(nx043));
    P2=NaN.*zeros(gpschan,length(nx043));
    P2std=NaN.*zeros(gpschan,length(nx043));
    L2=NaN.*zeros(gpschan,length(nx043));
    L2std=NaN.*zeros(gpschan,length(nx043));
    D2=NaN.*zeros(gpschan,length(nx043));
    CNo2=NaN.*zeros(gpschan,length(nx043));
    locktime2=NaN.*zeros(gpschan,length(nx043));
    parity2=NaN.*zeros(gpschan,length(nx043));
    P2type=NaN.*zeros(gpschan,length(nx043));
    
%     % GPS L5
%     PRN_rge5=NaN.*zeros(gpschan,length(nx043));
%     C5=NaN.*zeros(gpschan,length(nx043));
%     L5=NaN.*zeros(gpschan,length(nx043));
%     D5=NaN.*zeros(gpschan,length(nx043));
%     CNo5=NaN.*zeros(gpschan,length(nx043));
%     locktime5=NaN.*zeros(gpschan,length(nx043));
%     parity5=NaN.*zeros(gpschan,length(nx043));
    
    % QZSS L1 C/A
    PRN1q=NaN.*zeros(qzsschan,length(nx043));
    C1q=NaN.*zeros(qzsschan,length(nx043));
    L1q=NaN.*zeros(qzsschan,length(nx043));
    D1q=NaN.*zeros(qzsschan,length(nx043));
    CNo1q=NaN.*zeros(qzsschan,length(nx043));
    locktime1q=NaN.*zeros(qzsschan,length(nx043));
   
    % QZSS L2C
    PRN2q=NaN.*zeros(qzsschan,length(nx043));
    C2q=NaN.*zeros(qzsschan,length(nx043));
    L2q=NaN.*zeros(qzsschan,length(nx043));
    D2q=NaN.*zeros(qzsschan,length(nx043));
    CNo2q=NaN.*zeros(qzsschan,length(nx043));
    locktime2q=NaN.*zeros(qzsschan,length(nx043));

    % SBAS L1
    PRN1g=NaN.*zeros(sbaschn,length(nx043));
    C1g=NaN.*zeros(sbaschn,length(nx043));
    L1g=NaN.*zeros(sbaschn,length(nx043));
    D1g=NaN.*zeros(sbaschn,length(nx043));
    CNo1g=NaN.*zeros(sbaschn,length(nx043));
    locktime1g=NaN.*zeros(sbaschn,length(nx043));
    parity1g=NaN.*zeros(sbaschn,length(nx043));
    
%     % SBAS L5
%     C5g=NaN.*zeros(sbaschn,length(nx043));
%     L5g=NaN.*zeros(sbaschn,length(nx043));
%     D5g=NaN.*zeros(sbaschn,length(nx043));
%     CNo5g=NaN.*zeros(sbaschn,length(nx043));
%     locktime5g=NaN.*zeros(sbaschn,length(nx043));
%     parity5g=NaN.*zeros(sbaschn,length(nx043));
    
    % GLONASS L1
    PRN_1glo=NaN.*zeros(glochan,length(nx043));
    C1glo=NaN.*zeros(glochan,length(nx043));
    D1glo=NaN.*zeros(glochan,length(nx043));
    L1glo=NaN.*zeros(glochan,length(nx043));
    CNo1glo=NaN.*zeros(glochan,length(nx043));
    
    % GLONASS L2
    PRN_2glo=NaN.*zeros(glochan,length(nx043));
    C2glo=NaN.*zeros(glochan,length(nx043));
    D2glo=NaN.*zeros(glochan,length(nx043));
    L2glo=NaN.*zeros(glochan,length(nx043));
    CNo2glo=NaN.*zeros(glochan,length(nx043));

    % Galileo E1
    PRN1e=NaN.*zeros(galchan,length(nx043));
    C1e=NaN.*zeros(galchan,length(nx043));
    L1e=NaN.*zeros(galchan,length(nx043));
    D1e=NaN.*zeros(galchan,length(nx043));
    CNo1e=NaN.*zeros(galchan,length(nx043));
    locktimee=NaN.*zeros(galchan,length(nx043));

%     % Galileo E5a (Q)
%     PRN5e=NaN.*zeros(galchan,length(nx043));
%     C5e=NaN.*zeros(galchan,length(nx043));
%     D5e=NaN.*zeros(galchan,length(nx043));
%     L5e=NaN.*zeros(galchan,length(nx043));
%     CNo5e=NaN.*zeros(galchan,length(nx043));
%     locktime5e=NaN.*zeros(galchan,length(nx043));

    % Galileo E5b (Q)
    PRN5bgal=NaN.*zeros(galchan,length(nx043));
    C5bgal=NaN.*zeros(galchan,length(nx043));
    D5bgal=NaN.*zeros(galchan,length(nx043));
    L5bgal=NaN.*zeros(galchan,length(nx043));
    CNo5bgal=NaN.*zeros(galchan,length(nx043));
    locktime5bgal=NaN.*zeros(galchan,length(nx043));
    
    % BeiDou B1I
    PRN1bei=NaN.*zeros(beichan,length(nx043));
    C1bei=NaN.*zeros(beichan,length(nx043));
    D1bei=NaN.*zeros(beichan,length(nx043));
    L1bei=NaN.*zeros(beichan,length(nx043));
    CNo1bei=NaN.*zeros(beichan,length(nx043));
    locktime1bei=NaN.*zeros(beichan,length(nx043));
    
    % BeiDou B2I
    PRN2bei=NaN.*zeros(beichan,length(nx043));
    C2bei=NaN.*zeros(beichan,length(nx043));
    D2bei=NaN.*zeros(beichan,length(nx043));
    L2bei=NaN.*zeros(beichan,length(nx043));
    CNo2bei=NaN.*zeros(beichan,length(nx043));
    locktime2bei=NaN.*zeros(beichan,length(nx043));
    
    % BeiDou B2B
    PRN2Bbei=NaN.*zeros(beichan,length(nx043));
    C2Bbei=NaN.*zeros(beichan,length(nx043));
    D2Bbei=NaN.*zeros(beichan,length(nx043));
    L2Bbei=NaN.*zeros(beichan,length(nx043));
    CNo2Bbei=NaN.*zeros(beichan,length(nx043));
    locktime2Bbei=NaN.*zeros(beichan,length(nx043));
    

    for k = double(sort(unique(obs043),'descend'))' % do this backwards so MATLAB has a good chance at only having to allocate memory once
        nx = find(obs043==k);
        for kk = k:-1:1
            trkstat = typecast(reshape([m.data(nx043(nx)+H+44+44*(kk-1))'; m.data(nx043(nx)+H+45+44*(kk-1))';...
                m.data(nx043(nx)+H+46+44*(kk-1))'; m.data(nx043(nx)+H+47+44*(kk-1))'],[],1),'uint32');
            frq = bitand(bitshift(trkstat,-21),31);
            ssys = bitand(bitshift(trkstat,-16),7);
            chn = double(bitand(bitshift(trkstat,-5),31))+1;
            for chnk = unique(chn)'
                nx1 = nx(ssys==0 & frq==0 & chn==chnk); % GPS L1 observations
                if ~isempty(nx1)
                    PRN_rge1(chnk,nx1) = typecast(reshape([m.data(nx043(nx1)+H+4+44*(kk-1))'; m.data(nx043(nx1)+H+5+44*(kk-1))'],[],1),'uint16');
                    C1(chnk,nx1) = typecast(reshape([m.data(nx043(nx1)+H+8+44*(kk-1))'; m.data(nx043(nx1)+H+9+44*(kk-1))';...
                        m.data(nx043(nx1)+H+10+44*(kk-1))'; m.data(nx043(nx1)+H+11+44*(kk-1))';...
                        m.data(nx043(nx1)+H+12+44*(kk-1))'; m.data(nx043(nx1)+H+13+44*(kk-1))';...
                        m.data(nx043(nx1)+H+14+44*(kk-1))'; m.data(nx043(nx1)+H+15+44*(kk-1))'],[],1),'double');
                    C1std(chnk,nx1) = typecast(reshape([m.data(nx043(nx1)+H+16+44*(kk-1))'; m.data(nx043(nx1)+H+17+44*(kk-1))';...
                        m.data(nx043(nx1)+H+18+44*(kk-1))'; m.data(nx043(nx1)+H+19+44*(kk-1))'],[],1),'single');
                    L1(chnk,nx1) = typecast(reshape([m.data(nx043(nx1)+H+20+44*(kk-1))'; m.data(nx043(nx1)+H+21+44*(kk-1))';...
                        m.data(nx043(nx1)+H+22+44*(kk-1))'; m.data(nx043(nx1)+H+23+44*(kk-1))';...
                        m.data(nx043(nx1)+H+24+44*(kk-1))'; m.data(nx043(nx1)+H+25+44*(kk-1))';...
                        m.data(nx043(nx1)+H+26+44*(kk-1))'; m.data(nx043(nx1)+H+27+44*(kk-1))'],[],1),'double');
                    L1std(chnk,nx1) = typecast(reshape([m.data(nx043(nx1)+H+28+44*(kk-1))'; m.data(nx043(nx1)+H+29+44*(kk-1))';...
                        m.data(nx043(nx1)+H+30+44*(kk-1))'; m.data(nx043(nx1)+H+31+44*(kk-1))'],[],1),'single');
                    D1(chnk,nx1) = typecast(reshape([m.data(nx043(nx1)+H+32+44*(kk-1))'; m.data(nx043(nx1)+H+33+44*(kk-1))';...
                        m.data(nx043(nx1)+H+34+44*(kk-1))'; m.data(nx043(nx1)+H+35+44*(kk-1))'],[],1),'single');
                    CNo1(chnk,nx1) = typecast(reshape([m.data(nx043(nx1)+H+36+44*(kk-1))'; m.data(nx043(nx1)+H+37+44*(kk-1))';...
                        m.data(nx043(nx1)+H+38+44*(kk-1))'; m.data(nx043(nx1)+H+39+44*(kk-1))'],[],1),'single');
                    locktime1(chnk,nx1) = typecast(reshape([m.data(nx043(nx1)+H+40+44*(kk-1))'; m.data(nx043(nx1)+H+41+44*(kk-1))';...
                        m.data(nx043(nx1)+H+42+44*(kk-1))'; m.data(nx043(nx1)+H+43+44*(kk-1))'],[],1),'single');
                    parity1(chnk,nx1) = bitget(trkstat(ssys==0 & frq==0 & chn==chnk),12);
                    satsys1(chnk,nx1) = bitand(bitshift(trkstat(ssys==0 & frq==0 & chn==chnk),-16),7);
                end
                nx2 = nx(ssys==0 & (frq==9 | frq == 5) & chn==chnk); % GPS L2 observations
                if ~isempty(nx2)
                    PRN_rge2(chnk,nx2) = typecast(reshape([m.data(nx043(nx2)+H+4+44*(kk-1))'; m.data(nx043(nx2)+H+5+44*(kk-1))'],[],1),'uint16');
                    P2(chnk,nx2) = typecast(reshape([m.data(nx043(nx2)+H+8+44*(kk-1))'; m.data(nx043(nx2)+H+9+44*(kk-1))';...
                        m.data(nx043(nx2)+H+10+44*(kk-1))'; m.data(nx043(nx2)+H+11+44*(kk-1))';...
                        m.data(nx043(nx2)+H+12+44*(kk-1))'; m.data(nx043(nx2)+H+13+44*(kk-1))';...
                        m.data(nx043(nx2)+H+14+44*(kk-1))'; m.data(nx043(nx2)+H+15+44*(kk-1))'],[],1),'double');
                    P2std(chnk,nx2) = typecast(reshape([m.data(nx043(nx2)+H+16+44*(kk-1))'; m.data(nx043(nx2)+H+17+44*(kk-1))';...
                        m.data(nx043(nx2)+H+18+44*(kk-1))'; m.data(nx043(nx2)+H+19+44*(kk-1))'],[],1),'single');
                    L2(chnk,nx2) = typecast(reshape([m.data(nx043(nx2)+H+20+44*(kk-1))'; m.data(nx043(nx2)+H+21+44*(kk-1))';...
                        m.data(nx043(nx2)+H+22+44*(kk-1))'; m.data(nx043(nx2)+H+23+44*(kk-1))';...
                        m.data(nx043(nx2)+H+24+44*(kk-1))'; m.data(nx043(nx2)+H+25+44*(kk-1))';...
                        m.data(nx043(nx2)+H+26+44*(kk-1))'; m.data(nx043(nx2)+H+27+44*(kk-1))'],[],1),'double');
                    L2std(chnk,nx2) = typecast(reshape([m.data(nx043(nx2)+H+28+44*(kk-1))'; m.data(nx043(nx2)+H+29+44*(kk-1))';...
                        m.data(nx043(nx2)+H+30+44*(kk-1))'; m.data(nx043(nx2)+H+31+44*(kk-1))'],[],1),'single');
                    D2(chnk,nx2) = typecast(reshape([m.data(nx043(nx2)+H+32+44*(kk-1))'; m.data(nx043(nx2)+H+33+44*(kk-1))';...
                        m.data(nx043(nx2)+H+34+44*(kk-1))'; m.data(nx043(nx2)+H+35+44*(kk-1))'],[],1),'single');
                    CNo2(chnk,nx2) = typecast(reshape([m.data(nx043(nx2)+H+36+44*(kk-1))'; m.data(nx043(nx2)+H+37+44*(kk-1))';...
                        m.data(nx043(nx2)+H+38+44*(kk-1))'; m.data(nx043(nx2)+H+39+44*(kk-1))'],[],1),'single');
                    locktime2(chnk,nx2) = typecast(reshape([m.data(nx043(nx2)+H+40+44*(kk-1))'; m.data(nx043(nx2)+H+41+44*(kk-1))';...
                        m.data(nx043(nx2)+H+42+44*(kk-1))'; m.data(nx043(nx2)+H+43+44*(kk-1))'],[],1),'single');
%                     parity2(chnk,nx2) = bitget(trkstat(ssys==0 & (frq==9 | frq == 5) & chn==chnk),12);
%                     P2type(chnk,nx2) = bitand(bitshift(trkstat(ssys==0 & (frq==9 | frq == 5) & chn==chnk),-23),7);
                end
%                 nx1 = nx(ssys==0 & frq==14 & chn==chnk); % GPS L5 observations
%                 if ~isempty(nx1)
%                     PRN_rge5(chnk,nx1) = typecast(reshape([m.data(nx043(nx1)+H+4+44*(kk-1))'; m.data(nx043(nx1)+H+5+44*(kk-1))'],[],1),'uint16');
%                     C5(chnk,nx1) = typecast(reshape([m.data(nx043(nx1)+H+8+44*(kk-1))'; m.data(nx043(nx1)+H+9+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+10+44*(kk-1))'; m.data(nx043(nx1)+H+11+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+12+44*(kk-1))'; m.data(nx043(nx1)+H+13+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+14+44*(kk-1))'; m.data(nx043(nx1)+H+15+44*(kk-1))'],[],1),'double');
% %                     C5std(chnk,nx1) = typecast(reshape([m.data(nx043(nx1)+H+16+44*(kk-1))'; m.data(nx043(nx1)+H+17+44*(kk-1))';...
% %                         m.data(nx043(nx1)+H+18+44*(kk-1))'; m.data(nx043(nx1)+H+19+44*(kk-1))'],[],1),'single');
%                     L5(chnk,nx1) = typecast(reshape([m.data(nx043(nx1)+H+20+44*(kk-1))'; m.data(nx043(nx1)+H+21+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+22+44*(kk-1))'; m.data(nx043(nx1)+H+23+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+24+44*(kk-1))'; m.data(nx043(nx1)+H+25+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+26+44*(kk-1))'; m.data(nx043(nx1)+H+27+44*(kk-1))'],[],1),'double');
% %                     L5std(chnk,nx1) = typecast(reshape([m.data(nx043(nx1)+H+28+44*(kk-1))'; m.data(nx043(nx1)+H+29+44*(kk-1))';...
% %                         m.data(nx043(nx1)+H+30+44*(kk-1))'; m.data(nx043(nx1)+H+31+44*(kk-1))'],[],1),'single');
% %                     D5(chnk,nx1) = typecast(reshape([m.data(nx043(nx1)+H+32+44*(kk-1))'; m.data(nx043(nx1)+H+33+44*(kk-1))';...
% %                         m.data(nx043(nx1)+H+34+44*(kk-1))'; m.data(nx043(nx1)+H+35+44*(kk-1))'],[],1),'single');
%                     CNo5(chnk,nx1) = typecast(reshape([m.data(nx043(nx1)+H+36+44*(kk-1))'; m.data(nx043(nx1)+H+37+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+38+44*(kk-1))'; m.data(nx043(nx1)+H+39+44*(kk-1))'],[],1),'single');
%                     locktime5(chnk,nx1) = typecast(reshape([m.data(nx043(nx1)+H+40+44*(kk-1))'; m.data(nx043(nx1)+H+41+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+42+44*(kk-1))'; m.data(nx043(nx1)+H+43+44*(kk-1))'],[],1),'single');
%                     parity5(chnk,nx1) = bitget(trkstat(ssys==0 & frq==14 & chn==chnk),12);
% %                     satsys5(chnk,nx1) = bitand(bitshift(trkstat(ssys==0 & frq==14 & chn==chnk),-16),7);
%                 end
                nx1 = nx(ssys==5 & frq==0 & chn==chnk); % QZSS L1 C/A observations
                if ~isempty(nx1) && parse_qzss
                    chnkq = chnk - qzssofst;
                    PRN1q(chnkq,nx1) = typecast(reshape([m.data(nx043(nx1)+H+4+44*(kk-1))'; m.data(nx043(nx1)+H+5+44*(kk-1))'],[],1),'uint16');
                    C1q(chnkq,nx1) = typecast(reshape([m.data(nx043(nx1)+H+8+44*(kk-1))'; m.data(nx043(nx1)+H+9+44*(kk-1))';...
                        m.data(nx043(nx1)+H+10+44*(kk-1))'; m.data(nx043(nx1)+H+11+44*(kk-1))';...
                        m.data(nx043(nx1)+H+12+44*(kk-1))'; m.data(nx043(nx1)+H+13+44*(kk-1))';...
                        m.data(nx043(nx1)+H+14+44*(kk-1))'; m.data(nx043(nx1)+H+15+44*(kk-1))'],[],1),'double');
%                     C1qstd(chnkq,nx1) = typecast(reshape([m.data(nx043(nx1)+H+16+44*(kk-1))'; m.data(nx043(nx1)+H+17+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+18+44*(kk-1))'; m.data(nx043(nx1)+H+19+44*(kk-1))'],[],1),'single');
                    L1q(chnkq,nx1) = typecast(reshape([m.data(nx043(nx1)+H+20+44*(kk-1))'; m.data(nx043(nx1)+H+21+44*(kk-1))';...
                        m.data(nx043(nx1)+H+22+44*(kk-1))'; m.data(nx043(nx1)+H+23+44*(kk-1))';...
                        m.data(nx043(nx1)+H+24+44*(kk-1))'; m.data(nx043(nx1)+H+25+44*(kk-1))';...
                        m.data(nx043(nx1)+H+26+44*(kk-1))'; m.data(nx043(nx1)+H+27+44*(kk-1))'],[],1),'double');
%                     L1qstd(chnkq,nx1) = typecast(reshape([m.data(nx043(nx1)+H+28+44*(kk-1))'; m.data(nx043(nx1)+H+29+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+30+44*(kk-1))'; m.data(nx043(nx1)+H+31+44*(kk-1))'],[],1),'single');
                    D1q(chnkq,nx1) = typecast(reshape([m.data(nx043(nx1)+H+32+44*(kk-1))'; m.data(nx043(nx1)+H+33+44*(kk-1))';...
                        m.data(nx043(nx1)+H+34+44*(kk-1))'; m.data(nx043(nx1)+H+35+44*(kk-1))'],[],1),'single');
                    CNo1q(chnkq,nx1) = typecast(reshape([m.data(nx043(nx1)+H+36+44*(kk-1))'; m.data(nx043(nx1)+H+37+44*(kk-1))';...
                        m.data(nx043(nx1)+H+38+44*(kk-1))'; m.data(nx043(nx1)+H+39+44*(kk-1))'],[],1),'single');
                    locktime1q(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+40+44*(kk-1))'; m.data(nx043(nx1)+H+41+44*(kk-1))';...
                        m.data(nx043(nx1)+H+42+44*(kk-1))'; m.data(nx043(nx1)+H+43+44*(kk-1))'],[],1),'single');
                    parity1q(chnkq,nx1) = bitget(trkstat(ssys==2 & frq==0 & chn==chnk),12);
%                     satsys1q(chnkq,nx1) = bitand(bitshift(trkstat(frq==0 & chn==chnk),-16),7);
                end
                nx1 = nx(ssys==5 & frq==17 & chn==chnk); % QZSS L2C (M) observations
                if ~isempty(nx1) && parse_qzss
                    chnkq = chnk - qzssofst;
                    PRN2q(chnkq,nx1) = typecast(reshape([m.data(nx043(nx1)+H+4+44*(kk-1))'; m.data(nx043(nx1)+H+5+44*(kk-1))'],[],1),'uint16');
                    C2q(chnkq,nx1) = typecast(reshape([m.data(nx043(nx1)+H+8+44*(kk-1))'; m.data(nx043(nx1)+H+9+44*(kk-1))';...
                        m.data(nx043(nx1)+H+10+44*(kk-1))'; m.data(nx043(nx1)+H+11+44*(kk-1))';...
                        m.data(nx043(nx1)+H+12+44*(kk-1))'; m.data(nx043(nx1)+H+13+44*(kk-1))';...
                        m.data(nx043(nx1)+H+14+44*(kk-1))'; m.data(nx043(nx1)+H+15+44*(kk-1))'],[],1),'double');
%                     C1qstd(chnkq,nx1) = typecast(reshape([m.data(nx043(nx1)+H+16+44*(kk-1))'; m.data(nx043(nx1)+H+17+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+18+44*(kk-1))'; m.data(nx043(nx1)+H+19+44*(kk-1))'],[],1),'single');
                    L2q(chnkq,nx1) = typecast(reshape([m.data(nx043(nx1)+H+20+44*(kk-1))'; m.data(nx043(nx1)+H+21+44*(kk-1))';...
                        m.data(nx043(nx1)+H+22+44*(kk-1))'; m.data(nx043(nx1)+H+23+44*(kk-1))';...
                        m.data(nx043(nx1)+H+24+44*(kk-1))'; m.data(nx043(nx1)+H+25+44*(kk-1))';...
                        m.data(nx043(nx1)+H+26+44*(kk-1))'; m.data(nx043(nx1)+H+27+44*(kk-1))'],[],1),'double');
%                     L1qstd(chnkq,nx1) = typecast(reshape([m.data(nx043(nx1)+H+28+44*(kk-1))'; m.data(nx043(nx1)+H+29+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+30+44*(kk-1))'; m.data(nx043(nx1)+H+31+44*(kk-1))'],[],1),'single');
                    D2q(chnkq,nx1) = typecast(reshape([m.data(nx043(nx1)+H+32+44*(kk-1))'; m.data(nx043(nx1)+H+33+44*(kk-1))';...
                        m.data(nx043(nx1)+H+34+44*(kk-1))'; m.data(nx043(nx1)+H+35+44*(kk-1))'],[],1),'single');
                    CNo2q(chnkq,nx1) = typecast(reshape([m.data(nx043(nx1)+H+36+44*(kk-1))'; m.data(nx043(nx1)+H+37+44*(kk-1))';...
                        m.data(nx043(nx1)+H+38+44*(kk-1))'; m.data(nx043(nx1)+H+39+44*(kk-1))'],[],1),'single');
                    locktime2q(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+40+44*(kk-1))'; m.data(nx043(nx1)+H+41+44*(kk-1))';...
                        m.data(nx043(nx1)+H+42+44*(kk-1))'; m.data(nx043(nx1)+H+43+44*(kk-1))'],[],1),'single');
                    parity2q(chnkq,nx1) = bitget(trkstat(ssys==2 & frq==0 & chn==chnk),12);
%                     satsys2q(chnkq,nx1) = bitand(bitshift(trkstat(frq==0 & chn==chnk),-16),7);
                end
                nx1 = nx(ssys==2 & frq==0 & chn==chnk); % SBAS L1 observations
                if ~isempty(nx1) && parse_sbas
                    chnkg = chnk - sbasofst;
                    PRN1g(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+4+44*(kk-1))'; m.data(nx043(nx1)+H+5+44*(kk-1))'],[],1),'uint16');
                    C1g(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+8+44*(kk-1))'; m.data(nx043(nx1)+H+9+44*(kk-1))';...
                        m.data(nx043(nx1)+H+10+44*(kk-1))'; m.data(nx043(nx1)+H+11+44*(kk-1))';...
                        m.data(nx043(nx1)+H+12+44*(kk-1))'; m.data(nx043(nx1)+H+13+44*(kk-1))';...
                        m.data(nx043(nx1)+H+14+44*(kk-1))'; m.data(nx043(nx1)+H+15+44*(kk-1))'],[],1),'double');
%                     C1gstd(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+16+44*(kk-1))'; m.data(nx043(nx1)+H+17+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+18+44*(kk-1))'; m.data(nx043(nx1)+H+19+44*(kk-1))'],[],1),'single');
                    L1g(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+20+44*(kk-1))'; m.data(nx043(nx1)+H+21+44*(kk-1))';...
                        m.data(nx043(nx1)+H+22+44*(kk-1))'; m.data(nx043(nx1)+H+23+44*(kk-1))';...
                        m.data(nx043(nx1)+H+24+44*(kk-1))'; m.data(nx043(nx1)+H+25+44*(kk-1))';...
                        m.data(nx043(nx1)+H+26+44*(kk-1))'; m.data(nx043(nx1)+H+27+44*(kk-1))'],[],1),'double');
%                     L1gstd(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+28+44*(kk-1))'; m.data(nx043(nx1)+H+29+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+30+44*(kk-1))'; m.data(nx043(nx1)+H+31+44*(kk-1))'],[],1),'single');
                    D1g(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+32+44*(kk-1))'; m.data(nx043(nx1)+H+33+44*(kk-1))';...
                        m.data(nx043(nx1)+H+34+44*(kk-1))'; m.data(nx043(nx1)+H+35+44*(kk-1))'],[],1),'single');
                    CNo1g(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+36+44*(kk-1))'; m.data(nx043(nx1)+H+37+44*(kk-1))';...
                        m.data(nx043(nx1)+H+38+44*(kk-1))'; m.data(nx043(nx1)+H+39+44*(kk-1))'],[],1),'single');
                    locktime1g(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+40+44*(kk-1))'; m.data(nx043(nx1)+H+41+44*(kk-1))';...
                        m.data(nx043(nx1)+H+42+44*(kk-1))'; m.data(nx043(nx1)+H+43+44*(kk-1))'],[],1),'single');
                    parity1g(chnkg,nx1) = bitget(trkstat(ssys==2 & frq==0 & chn==chnk),12);
%                     satsys1g(chnkg,nx1) = bitand(bitshift(trkstat(frq==0 & chn==chnk),-16),7);
                end
%                 nx1 = nx(ssys==2 & frq==6 & chn==chnk); % SBAS L5 observations
%                 if ~isempty(nx1) && parse_sbas
%                     chnkg = chnk - sbasofst;
%                     PRN5g(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+4+44*(kk-1))'; m.data(nx043(nx1)+H+5+44*(kk-1))'],[],1),'uint16');
%                     C5g(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+8+44*(kk-1))'; m.data(nx043(nx1)+H+9+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+10+44*(kk-1))'; m.data(nx043(nx1)+H+11+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+12+44*(kk-1))'; m.data(nx043(nx1)+H+13+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+14+44*(kk-1))'; m.data(nx043(nx1)+H+15+44*(kk-1))'],[],1),'double');
% %                     C5gstd(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+16+44*(kk-1))'; m.data(nx043(nx1)+H+17+44*(kk-1))';...
% %                         m.data(nx043(nx1)+H+18+44*(kk-1))'; m.data(nx043(nx1)+H+19+44*(kk-1))'],[],1),'single');
%                     L5g(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+20+44*(kk-1))'; m.data(nx043(nx1)+H+21+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+22+44*(kk-1))'; m.data(nx043(nx1)+H+23+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+24+44*(kk-1))'; m.data(nx043(nx1)+H+25+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+26+44*(kk-1))'; m.data(nx043(nx1)+H+27+44*(kk-1))'],[],1),'double');
% %                     L5gstd(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+28+44*(kk-1))'; m.data(nx043(nx1)+H+29+44*(kk-1))';...
% %                         m.data(nx043(nx1)+H+30+44*(kk-1))'; m.data(nx043(nx1)+H+31+44*(kk-1))'],[],1),'single');
%                     D5g(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+32+44*(kk-1))'; m.data(nx043(nx1)+H+33+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+34+44*(kk-1))'; m.data(nx043(nx1)+H+35+44*(kk-1))'],[],1),'single');
%                     CNo5g(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+36+44*(kk-1))'; m.data(nx043(nx1)+H+37+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+38+44*(kk-1))'; m.data(nx043(nx1)+H+39+44*(kk-1))'],[],1),'single');
%                     locktime5g(chnkg,nx1) = typecast(reshape([m.data(nx043(nx1)+H+40+44*(kk-1))'; m.data(nx043(nx1)+H+41+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+42+44*(kk-1))'; m.data(nx043(nx1)+H+43+44*(kk-1))'],[],1),'single');
%                     parity5g(chnkg,nx1) = bitget(trkstat(ssys==2 & frq==6 & chn==chnk),12);
% %                     satsys5g(chnk,nx1) = bitand(bitshift(trkstat(frq==0 & chn==chnk),-16),7);
%                 end
                nx1 = nx(ssys==1 & frq==0 & chn==chnk); % GLONASS L1 observations
                if ~isempty(nx1) && parse_glo
                    if chnk > 6
                        chnkglo = chnk - gloofst2;
                    else
                        chnkglo = chnk - gloofst1;
                    end
                    PRN1glo(chnkglo,nx1) = typecast(reshape([m.data(nx043(nx1)+H+4+44*(kk-1))'; m.data(nx043(nx1)+H+5+44*(kk-1))'],[],1),'uint16');
                    C1glo(chnkglo,nx1) = typecast(reshape([m.data(nx043(nx1)+H+8+44*(kk-1))'; m.data(nx043(nx1)+H+9+44*(kk-1))';...
                        m.data(nx043(nx1)+H+10+44*(kk-1))'; m.data(nx043(nx1)+H+11+44*(kk-1))';...
                        m.data(nx043(nx1)+H+12+44*(kk-1))'; m.data(nx043(nx1)+H+13+44*(kk-1))';...
                        m.data(nx043(nx1)+H+14+44*(kk-1))'; m.data(nx043(nx1)+H+15+44*(kk-1))'],[],1),'double'); 
                    L1glo(chnkglo,nx1) = typecast(reshape([m.data(nx043(nx1)+H+20+44*(kk-1))'; m.data(nx043(nx1)+H+21+44*(kk-1))';...
                        m.data(nx043(nx1)+H+22+44*(kk-1))'; m.data(nx043(nx1)+H+23+44*(kk-1))';...
                        m.data(nx043(nx1)+H+24+44*(kk-1))'; m.data(nx043(nx1)+H+25+44*(kk-1))';...
                        m.data(nx043(nx1)+H+26+44*(kk-1))'; m.data(nx043(nx1)+H+27+44*(kk-1))'],[],1),'double');
                    D1glo(chnkglo,nx1) = typecast(reshape([m.data(nx043(nx1)+H+32+44*(kk-1))'; m.data(nx043(nx1)+H+33+44*(kk-1))';...
                        m.data(nx043(nx1)+H+34+44*(kk-1))'; m.data(nx043(nx1)+H+35+44*(kk-1))'],[],1),'single');
                    CNo1glo(chnkglo,nx1) = typecast(reshape([m.data(nx043(nx1)+H+36+44*(kk-1))'; m.data(nx043(nx1)+H+37+44*(kk-1))';...
                        m.data(nx043(nx1)+H+38+44*(kk-1))'; m.data(nx043(nx1)+H+39+44*(kk-1))'],[],1),'single');
                    locktime1glo(chnkglo,nx1) = typecast(reshape([m.data(nx043(nx1)+H+40+44*(kk-1))'; m.data(nx043(nx1)+H+41+44*(kk-1))';...
                        m.data(nx043(nx1)+H+42+44*(kk-1))'; m.data(nx043(nx1)+H+43+44*(kk-1))'],[],1),'single');
                end
                nx1 = nx(ssys==1 & frq==5 & chn==chnk); % GLONASS L2 observations
                if ~isempty(nx1) && parse_glo
                    if chnk > 6
                        chnkglo = chnk - gloofst2;
                    else
                        chnkglo = chnk - gloofst1;
                    end
                    PRN2glo(chnkglo,nx1) = typecast(reshape([m.data(nx043(nx1)+H+4+44*(kk-1))'; m.data(nx043(nx1)+H+5+44*(kk-1))'],[],1),'uint16');
                    C2glo(chnkglo,nx1) = typecast(reshape([m.data(nx043(nx1)+H+8+44*(kk-1))'; m.data(nx043(nx1)+H+9+44*(kk-1))';...
                        m.data(nx043(nx1)+H+10+44*(kk-1))'; m.data(nx043(nx1)+H+11+44*(kk-1))';...
                        m.data(nx043(nx1)+H+12+44*(kk-1))'; m.data(nx043(nx1)+H+13+44*(kk-1))';...
                        m.data(nx043(nx1)+H+14+44*(kk-1))'; m.data(nx043(nx1)+H+15+44*(kk-1))'],[],1),'double'); 
                    L2glo(chnkglo,nx1) = typecast(reshape([m.data(nx043(nx1)+H+20+44*(kk-1))'; m.data(nx043(nx1)+H+21+44*(kk-1))';...
                        m.data(nx043(nx1)+H+22+44*(kk-1))'; m.data(nx043(nx1)+H+23+44*(kk-1))';...
                        m.data(nx043(nx1)+H+24+44*(kk-1))'; m.data(nx043(nx1)+H+25+44*(kk-1))';...
                        m.data(nx043(nx1)+H+26+44*(kk-1))'; m.data(nx043(nx1)+H+27+44*(kk-1))'],[],1),'double');
                    D2glo(chnkglo,nx1) = typecast(reshape([m.data(nx043(nx1)+H+32+44*(kk-1))'; m.data(nx043(nx1)+H+33+44*(kk-1))';...
                        m.data(nx043(nx1)+H+34+44*(kk-1))'; m.data(nx043(nx1)+H+35+44*(kk-1))'],[],1),'single');
                    CNo2glo(chnkglo,nx1) = typecast(reshape([m.data(nx043(nx1)+H+36+44*(kk-1))'; m.data(nx043(nx1)+H+37+44*(kk-1))';...
                        m.data(nx043(nx1)+H+38+44*(kk-1))'; m.data(nx043(nx1)+H+39+44*(kk-1))'],[],1),'single');
                    locktime2glo(chnkglo,nx1) = typecast(reshape([m.data(nx043(nx1)+H+40+44*(kk-1))'; m.data(nx043(nx1)+H+41+44*(kk-1))';...
                        m.data(nx043(nx1)+H+42+44*(kk-1))'; m.data(nx043(nx1)+H+43+44*(kk-1))'],[],1),'single');
                end
                nx1 = nx(ssys==3 & frq==2 & chn==chnk); % Galileo L1 (E1C) observations
                if ~isempty(nx1) && parse_gal
                    chnke = chnk - galofst;
                    PRN1e(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+4+44*(kk-1))'; m.data(nx043(nx1)+H+5+44*(kk-1))'],[],1),'uint16');
                    C1e(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+8+44*(kk-1))'; m.data(nx043(nx1)+H+9+44*(kk-1))';...
                        m.data(nx043(nx1)+H+10+44*(kk-1))'; m.data(nx043(nx1)+H+11+44*(kk-1))';...
                        m.data(nx043(nx1)+H+12+44*(kk-1))'; m.data(nx043(nx1)+H+13+44*(kk-1))';...
                        m.data(nx043(nx1)+H+14+44*(kk-1))'; m.data(nx043(nx1)+H+15+44*(kk-1))'],[],1),'double');
%                     C1estd(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+16+44*(kk-1))'; m.data(nx043(nx1)+H+17+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+18+44*(kk-1))'; m.data(nx043(nx1)+H+19+44*(kk-1))'],[],1),'single');
                    L1e(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+20+44*(kk-1))'; m.data(nx043(nx1)+H+21+44*(kk-1))';...
                        m.data(nx043(nx1)+H+22+44*(kk-1))'; m.data(nx043(nx1)+H+23+44*(kk-1))';...
                        m.data(nx043(nx1)+H+24+44*(kk-1))'; m.data(nx043(nx1)+H+25+44*(kk-1))';...
                        m.data(nx043(nx1)+H+26+44*(kk-1))'; m.data(nx043(nx1)+H+27+44*(kk-1))'],[],1),'double');
%                     L1stde(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+28+44*(kk-1))'; m.data(nx043(nx1)+H+29+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+30+44*(kk-1))'; m.data(nx043(nx1)+H+31+44*(kk-1))'],[],1),'single');
                    D1e(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+32+44*(kk-1))'; m.data(nx043(nx1)+H+33+44*(kk-1))';...
                        m.data(nx043(nx1)+H+34+44*(kk-1))'; m.data(nx043(nx1)+H+35+44*(kk-1))'],[],1),'single');
                    CNo1e(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+36+44*(kk-1))'; m.data(nx043(nx1)+H+37+44*(kk-1))';...
                        m.data(nx043(nx1)+H+38+44*(kk-1))'; m.data(nx043(nx1)+H+39+44*(kk-1))'],[],1),'single');
%                     locktime1e(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+40+44*(kk-1))'; m.data(nx043(nx1)+H+41+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+42+44*(kk-1))'; m.data(nx043(nx1)+H+43+44*(kk-1))'],[],1),'single');
%                     parity1e(chnke,nx1) = bitget(trkstat(frq==0 & chn==chnk),12);
%                     satsys1e(chnke,nx1) = bitand(bitshift(trkstat(frq==0 & chn==chnk),-16),7);
                end
%                 nx1 = nx(ssys==3 & frq==12 & chn==chnk); % Galileo L5 (E5a Q) observations
%                 if ~isempty(nx1) && parse_gal
%                     chnke = chnk - galofst;
%                     PRN5e(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+4+44*(kk-1))'; m.data(nx043(nx1)+H+5+44*(kk-1))'],[],1),'uint16');
%                     C5e(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+8+44*(kk-1))'; m.data(nx043(nx1)+H+9+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+10+44*(kk-1))'; m.data(nx043(nx1)+H+11+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+12+44*(kk-1))'; m.data(nx043(nx1)+H+13+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+14+44*(kk-1))'; m.data(nx043(nx1)+H+15+44*(kk-1))'],[],1),'double');
% %                     C5estd(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+16+44*(kk-1))'; m.data(nx043(nx1)+H+17+44*(kk-1))';...
% %                         m.data(nx043(nx1)+H+18+44*(kk-1))'; m.data(nx043(nx1)+H+19+44*(kk-1))'],[],1),'single');
%                     L5e(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+20+44*(kk-1))'; m.data(nx043(nx1)+H+21+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+22+44*(kk-1))'; m.data(nx043(nx1)+H+23+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+24+44*(kk-1))'; m.data(nx043(nx1)+H+25+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+26+44*(kk-1))'; m.data(nx043(nx1)+H+27+44*(kk-1))'],[],1),'double');
% %                     L5estd(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+28+44*(kk-1))'; m.data(nx043(nx1)+H+29+44*(kk-1))';...
% %                         m.data(nx043(nx1)+H+30+44*(kk-1))'; m.data(nx043(nx1)+H+31+44*(kk-1))'],[],1),'single');
%                     D5e(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+32+44*(kk-1))'; m.data(nx043(nx1)+H+33+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+34+44*(kk-1))'; m.data(nx043(nx1)+H+35+44*(kk-1))'],[],1),'single');
%                     CNo5e(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+36+44*(kk-1))'; m.data(nx043(nx1)+H+37+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+38+44*(kk-1))'; m.data(nx043(nx1)+H+39+44*(kk-1))'],[],1),'single');
%                     locktime5e(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+40+44*(kk-1))'; m.data(nx043(nx1)+H+41+44*(kk-1))';...
%                         m.data(nx043(nx1)+H+42+44*(kk-1))'; m.data(nx043(nx1)+H+43+44*(kk-1))'],[],1),'single');
% %                     parity5e(chnke,nx1) = bitget(trkstat(frq==0 & chn==chnk),12);
% %                     satsys5e(chnke,nx1) = bitand(bitshift(trkstat(frq==0 & chn==chnk),-16),7);
%                 end
                nx1 = nx(ssys==3 & frq==17 & chn==chnk); % Galileo L5 (E5b Q) observations
                if ~isempty(nx1) && parse_gal
                    chnke = chnk - galofst;
                    PRN5bgal(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+4+44*(kk-1))'; m.data(nx043(nx1)+H+5+44*(kk-1))'],[],1),'uint16');
                    C5bgal(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+8+44*(kk-1))'; m.data(nx043(nx1)+H+9+44*(kk-1))';...
                        m.data(nx043(nx1)+H+10+44*(kk-1))'; m.data(nx043(nx1)+H+11+44*(kk-1))';...
                        m.data(nx043(nx1)+H+12+44*(kk-1))'; m.data(nx043(nx1)+H+13+44*(kk-1))';...
                        m.data(nx043(nx1)+H+14+44*(kk-1))'; m.data(nx043(nx1)+H+15+44*(kk-1))'],[],1),'double'); 
                    L5bgal(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+20+44*(kk-1))'; m.data(nx043(nx1)+H+21+44*(kk-1))';...
                        m.data(nx043(nx1)+H+22+44*(kk-1))'; m.data(nx043(nx1)+H+23+44*(kk-1))';...
                        m.data(nx043(nx1)+H+24+44*(kk-1))'; m.data(nx043(nx1)+H+25+44*(kk-1))';...
                        m.data(nx043(nx1)+H+26+44*(kk-1))'; m.data(nx043(nx1)+H+27+44*(kk-1))'],[],1),'double');
                    D5bgal(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+32+44*(kk-1))'; m.data(nx043(nx1)+H+33+44*(kk-1))';...
                        m.data(nx043(nx1)+H+34+44*(kk-1))'; m.data(nx043(nx1)+H+35+44*(kk-1))'],[],1),'single');
                    CNo5bgal(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+36+44*(kk-1))'; m.data(nx043(nx1)+H+37+44*(kk-1))';...
                        m.data(nx043(nx1)+H+38+44*(kk-1))'; m.data(nx043(nx1)+H+39+44*(kk-1))'],[],1),'single');
                    locktime5bgal(chnke,nx1) = typecast(reshape([m.data(nx043(nx1)+H+40+44*(kk-1))'; m.data(nx043(nx1)+H+41+44*(kk-1))';...
                        m.data(nx043(nx1)+H+42+44*(kk-1))'; m.data(nx043(nx1)+H+43+44*(kk-1))'],[],1),'single');
                end
                nx1 = nx(ssys==4 & (frq==0 | frq==4) & chn==chnk); % BeiDou B1 (I) D1/D2 data observations
                if ~isempty(nx1) && parse_bei
                    if chnk > 12
                        chnkbei = chnk - beiofst2;
                    else
                        chnkbei = chnk - beiofst1;
                    end
                    PRN1bei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+4+44*(kk-1))'; m.data(nx043(nx1)+H+5+44*(kk-1))'],[],1),'uint16');
                    C1bei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+8+44*(kk-1))'; m.data(nx043(nx1)+H+9+44*(kk-1))';...
                        m.data(nx043(nx1)+H+10+44*(kk-1))'; m.data(nx043(nx1)+H+11+44*(kk-1))';...
                        m.data(nx043(nx1)+H+12+44*(kk-1))'; m.data(nx043(nx1)+H+13+44*(kk-1))';...
                        m.data(nx043(nx1)+H+14+44*(kk-1))'; m.data(nx043(nx1)+H+15+44*(kk-1))'],[],1),'double'); 
                    L1bei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+20+44*(kk-1))'; m.data(nx043(nx1)+H+21+44*(kk-1))';...
                        m.data(nx043(nx1)+H+22+44*(kk-1))'; m.data(nx043(nx1)+H+23+44*(kk-1))';...
                        m.data(nx043(nx1)+H+24+44*(kk-1))'; m.data(nx043(nx1)+H+25+44*(kk-1))';...
                        m.data(nx043(nx1)+H+26+44*(kk-1))'; m.data(nx043(nx1)+H+27+44*(kk-1))'],[],1),'double');
                    D1bei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+32+44*(kk-1))'; m.data(nx043(nx1)+H+33+44*(kk-1))';...
                        m.data(nx043(nx1)+H+34+44*(kk-1))'; m.data(nx043(nx1)+H+35+44*(kk-1))'],[],1),'single');
                    CNo1bei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+36+44*(kk-1))'; m.data(nx043(nx1)+H+37+44*(kk-1))';...
                        m.data(nx043(nx1)+H+38+44*(kk-1))'; m.data(nx043(nx1)+H+39+44*(kk-1))'],[],1),'single');
                    locktime1bei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+40+44*(kk-1))'; m.data(nx043(nx1)+H+41+44*(kk-1))';...
                        m.data(nx043(nx1)+H+42+44*(kk-1))'; m.data(nx043(nx1)+H+43+44*(kk-1))'],[],1),'single');
                end
                nx1 = nx(ssys==4 & (frq==1 | frq==5) & chn==chnk); % BeiDou B2 (I) D1/D2 data observations
                if ~isempty(nx1) && parse_bei
                    if chnk > 12
                        chnkbei = chnk - beiofst2;
                    else
                        chnkbei = chnk - beiofst1;
                    end
                    PRN2bei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+4+44*(kk-1))'; m.data(nx043(nx1)+H+5+44*(kk-1))'],[],1),'uint16');
                    C2bei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+8+44*(kk-1))'; m.data(nx043(nx1)+H+9+44*(kk-1))';...
                        m.data(nx043(nx1)+H+10+44*(kk-1))'; m.data(nx043(nx1)+H+11+44*(kk-1))';...
                        m.data(nx043(nx1)+H+12+44*(kk-1))'; m.data(nx043(nx1)+H+13+44*(kk-1))';...
                        m.data(nx043(nx1)+H+14+44*(kk-1))'; m.data(nx043(nx1)+H+15+44*(kk-1))'],[],1),'double'); 
                    L2bei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+20+44*(kk-1))'; m.data(nx043(nx1)+H+21+44*(kk-1))';...
                        m.data(nx043(nx1)+H+22+44*(kk-1))'; m.data(nx043(nx1)+H+23+44*(kk-1))';...
                        m.data(nx043(nx1)+H+24+44*(kk-1))'; m.data(nx043(nx1)+H+25+44*(kk-1))';...
                        m.data(nx043(nx1)+H+26+44*(kk-1))'; m.data(nx043(nx1)+H+27+44*(kk-1))'],[],1),'double');
                    D2bei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+32+44*(kk-1))'; m.data(nx043(nx1)+H+33+44*(kk-1))';...
                        m.data(nx043(nx1)+H+34+44*(kk-1))'; m.data(nx043(nx1)+H+35+44*(kk-1))'],[],1),'single');
                    CNo2bei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+36+44*(kk-1))'; m.data(nx043(nx1)+H+37+44*(kk-1))';...
                        m.data(nx043(nx1)+H+38+44*(kk-1))'; m.data(nx043(nx1)+H+39+44*(kk-1))'],[],1),'single');
                    locktime2bei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+40+44*(kk-1))'; m.data(nx043(nx1)+H+41+44*(kk-1))';...
                        m.data(nx043(nx1)+H+42+44*(kk-1))'; m.data(nx043(nx1)+H+43+44*(kk-1))'],[],1),'single');
                end
                nx1 = nx(ssys==4 & frq==11 & chn==chnk); % BeiDou B2b data observations
                if ~isempty(nx1) && parse_bei
                    if chnk > 12
                        chnkbei = chnk - beiofst2;
                    else
                        chnkbei = chnk - beiofst1;
                    end
                    PRN2Bbei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+4+44*(kk-1))'; m.data(nx043(nx1)+H+5+44*(kk-1))'],[],1),'uint16');
                    C2Bbei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+8+44*(kk-1))'; m.data(nx043(nx1)+H+9+44*(kk-1))';...
                        m.data(nx043(nx1)+H+10+44*(kk-1))'; m.data(nx043(nx1)+H+11+44*(kk-1))';...
                        m.data(nx043(nx1)+H+12+44*(kk-1))'; m.data(nx043(nx1)+H+13+44*(kk-1))';...
                        m.data(nx043(nx1)+H+14+44*(kk-1))'; m.data(nx043(nx1)+H+15+44*(kk-1))'],[],1),'double'); 
                    L2Bbei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+20+44*(kk-1))'; m.data(nx043(nx1)+H+21+44*(kk-1))';...
                        m.data(nx043(nx1)+H+22+44*(kk-1))'; m.data(nx043(nx1)+H+23+44*(kk-1))';...
                        m.data(nx043(nx1)+H+24+44*(kk-1))'; m.data(nx043(nx1)+H+25+44*(kk-1))';...
                        m.data(nx043(nx1)+H+26+44*(kk-1))'; m.data(nx043(nx1)+H+27+44*(kk-1))'],[],1),'double');
                    D2Bbei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+32+44*(kk-1))'; m.data(nx043(nx1)+H+33+44*(kk-1))';...
                        m.data(nx043(nx1)+H+34+44*(kk-1))'; m.data(nx043(nx1)+H+35+44*(kk-1))'],[],1),'single');
                    CNo2Bbei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+36+44*(kk-1))'; m.data(nx043(nx1)+H+37+44*(kk-1))';...
                        m.data(nx043(nx1)+H+38+44*(kk-1))'; m.data(nx043(nx1)+H+39+44*(kk-1))'],[],1),'single');
                    locktime2Bbei(chnkbei,nx1) = typecast(reshape([m.data(nx043(nx1)+H+40+44*(kk-1))'; m.data(nx043(nx1)+H+41+44*(kk-1))';...
                        m.data(nx043(nx1)+H+42+44*(kk-1))'; m.data(nx043(nx1)+H+43+44*(kk-1))'],[],1),'single');
                end
            end
        end
    end
end

%%
% get AGCSTATS log index, ID:630, or uint8([118 2])
nx630 = nxrec(m.data(nxrec+4)==118 & m.data(nxrec+5)==2);
if ~isempty(nx630)
    fprintf('AGCSTATS...')
    msgln630 = typecast(reshape([m.data(nx630+8)'; m.data(nx630+9)'],[],1),'uint16');
    ms630 = typecast(reshape([m.data(nx630+16)'; m.data(nx630+17)'; m.data(nx630+18)'; m.data(nx630+19)'],[],1),'uint32');
    sow_agc = double(ms630)./1000; % timestamps for sqmdata
    obs630 = typecast(reshape([m.data(nx630+H)'; m.data(nx630+H+1)';...
        m.data(nx630+H+2)'; m.data(nx630+H+3)'],[],1),'uint32');
    nx630 = nx630(obs630==nanmedian(obs630)); % should all be the same, but just to make sure.
    jammer_L1 = NaN.*zeros(size(nx630));
    jammer_L2 = NaN.*zeros(size(nx630));
    jammer_L5 = NaN.*zeros(size(nx630));
    agc_bit_L1 = NaN.*zeros(size(nx630));
    agc_bit_L2 = NaN.*zeros(size(nx630));
    agc_bit_L5 = NaN.*zeros(size(nx630));
    agc_gain_L1 = NaN.*zeros(size(nx630));
    agc_gain_L2 = NaN.*zeros(size(nx630));
    agc_gain_L5 = NaN.*zeros(size(nx630));
    agc_bins_L1 = NaN.*zeros(6,length(nx630));
    agc_bins_L2 = NaN.*zeros(6,length(nx630));
    agc_bins_L5 = NaN.*zeros(6,length(nx630));

    for k = 1:double(nanmedian(obs630))
        AGCword = typecast(reshape([m.data(nx630+H+4+(k-1)*88)'; m.data(nx630+H+5+(k-1)*88)'; m.data(nx630+H+6+(k-1)*88)'; m.data(nx630+H+7+(k-1)*88)'],[],1),'uint32');
        rftype = bitand(bitshift(AGCword,-3),7);
        nx = find(rftype == 1); % L1 agc data
        if ~isempty(nx)
            jammer_L1(nx) = double(bitget(AGCword(nx),1));
            agc_bit_L1(nx) = double(bitand(bitshift(AGCword(nx),-1),3));
            agc_gain_L1(nx) = double(typecast(reshape([m.data(nx630(nx)+H+8+(k-1)*88)'; m.data(nx630(nx)+H+9+(k-1)*88)'; m.data(nx630(nx)+H+10+(k-1)*88)'; m.data(nx630(nx)+H+11+(k-1)*88)'],[],1),'uint32'));
            for kk = 6:-1:1
                agc_bins_L1(kk,nx) = typecast(reshape([m.data(nx630(nx)+H+kk*8+12+(k-1)*88)'; m.data(nx630(nx)+H+kk*8+13+(k-1)*88)';...
                    m.data(nx630(nx)+H+kk*8+14+(k-1)*88)'; m.data(nx630(nx)+H+kk*8+15+(k-1)*88)';...
                    m.data(nx630(nx)+H+kk*8+16+(k-1)*88)'; m.data(nx630(nx)+H+kk*8+17+(k-1)*88)';...
                    m.data(nx630(nx)+H+kk*8+18+(k-1)*88)'; m.data(nx630(nx)+H+kk*8+19+(k-1)*88)'],[],1),'double'); 
            end
        end
        nx = find(rftype == 2); % L2 agc data
        if ~isempty(nx)
            jammer_L2(nx) = double(bitget(AGCword(nx),1));
            agc_bit_L2(nx) = double(bitand(bitshift(AGCword(nx),-1),3));
            agc_gain_L2(nx) = double(typecast(reshape([m.data(nx630(nx)+H+8+(k-1)*88)'; m.data(nx630(nx)+H+9+(k-1)*88)'; m.data(nx630(nx)+H+10+(k-1)*88)'; m.data(nx630(nx)+H+11+(k-1)*88)'],[],1),'uint32'));
            for kk = 6:-1:1
                agc_bins_L2(kk,nx) = typecast(reshape([m.data(nx630(nx)+H+kk*8+12+(k-1)*88)'; m.data(nx630(nx)+H+kk*8+13+(k-1)*88)';...
                    m.data(nx630(nx)+H+kk*8+14+(k-1)*88)'; m.data(nx630(nx)+H+kk*8+15+(k-1)*88)';...
                    m.data(nx630(nx)+H+kk*8+16+(k-1)*88)'; m.data(nx630(nx)+H+kk*8+17+(k-1)*88)';...
                    m.data(nx630(nx)+H+kk*8+18+(k-1)*88)'; m.data(nx630(nx)+H+kk*8+19+(k-1)*88)'],[],1),'double'); 
            end
        end
        nx = find(rftype == 3); % L5 agc data
        if ~isempty(nx)
            jammer_L5(nx) = double(bitget(AGCword(nx),1));
            agc_bit_L5(nx) = double(bitand(bitshift(AGCword(nx),-1),3));
            agc_gain_L5(nx) = double(typecast(reshape([m.data(nx630(nx)+H+8+(k-1)*88)'; m.data(nx630(nx)+H+9+(k-1)*88)'; m.data(nx630(nx)+H+10+(k-1)*88)'; m.data(nx630(nx)+H+11+(k-1)*88)'],[],1),'uint32'));
            for kk = 6:-1:1
                agc_bins_L5(kk,nx) = typecast(reshape([m.data(nx630(nx)+H+kk*8+12+(k-1)*88)'; m.data(nx630(nx)+H+kk*8+13+(k-1)*88)';...
                    m.data(nx630(nx)+H+kk*8+14+(k-1)*88)'; m.data(nx630(nx)+H+kk*8+15+(k-1)*88)';...
                    m.data(nx630(nx)+H+kk*8+16+(k-1)*88)'; m.data(nx630(nx)+H+kk*8+17+(k-1)*88)';...
                    m.data(nx630(nx)+H+kk*8+18+(k-1)*88)'; m.data(nx630(nx)+H+kk*8+19+(k-1)*88)'],[],1),'double'); 
            end
        end
    end
    nxeow = find(sow_agc(2:end)-sow_agc(1:end-1)<-604797);
    while ~isempty(nxeow)
        for k = 1:length(nxeow)
            sow_agc(nxeow(k)+1:end) = sow_agc(nxeow(k)+1:end)+604800;
        end
        nxeow = find(sow_agc(2:end)-sow_agc(1:end-1)<-604797);
    end
end

%%
% get SYSTEMLEVELS log index, ID:653, or uint8([141 2])
nx653 = nxrec(m.data(nxrec+4)==141 & m.data(nxrec+5)==2);
if ~isempty(nx653)
    fprintf('temperature...')
    msgln653 = typecast(reshape([m.data(nx653+8)'; m.data(nx653+9)'],[],1),'uint16');
    ms653 = typecast(reshape([m.data(nx653+16)'; m.data(nx653+17)'; m.data(nx653+18)'; m.data(nx653+19)'],[],1),'uint32');
    sowtemp = double(ms653)./1000; % timestamps for sqmdata
    obs653 = typecast(reshape([m.data(nx653+H)'; m.data(nx653+H+1)';...
        m.data(nx653+H+2)'; m.data(nx653+H+3)'],[],1),'uint32');
    obs1x = nx653(obs653>0); % first grab data from SYSTEMLEVELS records with observations from at least 1 component
    nx1x = find(obs653>0);

    cmp = 0;
    tmp0 = typecast(reshape([m.data(nx653+H+12+(cmp*48)+0)'; m.data(nx653+H+12+(cmp*48)+1)';m.data(nx653+H+12+(cmp*48)+2)'; m.data(nx653+H+12+(cmp*48)+3)'],[],1),'single');
    idl0 = typecast(reshape([m.data(nx653+H+44+(cmp*48)+0)'; m.data(nx653+H+44+(cmp*48)+1)';m.data(nx653+H+44+(cmp*48)+2)'; m.data(nx653+H+44+(cmp*48)+3)'],[],1),'single');
    involt0 = typecast(reshape([m.data(nx653+H+24+(cmp*48)+0)'; m.data(nx653+H+24+(cmp*48)+1)';m.data(nx653+H+24+(cmp*48)+2)'; m.data(nx653+H+24+(cmp*48)+3)'],[],1),'single');
    lnavolt0 = typecast(reshape([m.data(nx653+H+32+(cmp*48)+0)'; m.data(nx653+H+32+(cmp*48)+1)';m.data(nx653+H+32+(cmp*48)+2)'; m.data(nx653+H+32+(cmp*48)+3)'],[],1),'single');
    cmp = 1;
    tmp1 = typecast(reshape([m.data(nx653+H+12+(cmp*48)+0)'; m.data(nx653+H+12+(cmp*48)+1)';m.data(nx653+H+12+(cmp*48)+2)'; m.data(nx653+H+12+(cmp*48)+3)'],[],1),'single');
    idl1 = typecast(reshape([m.data(nx653+H+44+(cmp*48)+0)'; m.data(nx653+H+44+(cmp*48)+1)';m.data(nx653+H+44+(cmp*48)+2)'; m.data(nx653+H+44+(cmp*48)+3)'],[],1),'single');
    lnavolt1 = typecast(reshape([m.data(nx653+H+32+(cmp*48)+0)'; m.data(nx653+H+32+(cmp*48)+1)';m.data(nx653+H+32+(cmp*48)+2)'; m.data(nx653+H+32+(cmp*48)+3)'],[],1),'single');
    cmp = 2;
    tmp2 = typecast(reshape([m.data(nx653+H+12+(cmp*48)+0)'; m.data(nx653+H+12+(cmp*48)+1)';m.data(nx653+H+12+(cmp*48)+2)'; m.data(nx653+H+12+(cmp*48)+3)'],[],1),'single');
    idl2 = typecast(reshape([m.data(nx653+H+44+(cmp*48)+0)'; m.data(nx653+H+44+(cmp*48)+1)';m.data(nx653+H+44+(cmp*48)+2)'; m.data(nx653+H+44+(cmp*48)+3)'],[],1),'single');
end

%%
% get ALLSQMI log index, ID:632, or uint8([120 2])
nxrx = find(m.data(nxrec+4)==120 & m.data(nxrec+5)==2);
if ~isempty(nxrx) && parse_allsqmib
    fprintf('ALLSQMI...')
    nx632 = nxrec(nxrx);
    obs632 = typecast(reshape([m.data(nx632+H)'; m.data(nx632+H+1)';...
        m.data(nx632+H+2)'; m.data(nx632+H+3)'],[],1),'uint32');
    nx632 = nx632(obs632>0 & obs632<100); % only taking records with a reasonable number of observations
    obs632 = typecast(reshape([m.data(nx632+H)'; m.data(nx632+H+1)';...
        m.data(nx632+H+2)'; m.data(nx632+H+3)'],[],1),'uint32');
    msgln632 = typecast(reshape([m.data(nx632+8)'; m.data(nx632+9)'],[],1),'uint16');
    ms632 = typecast(reshape([m.data(nx632+16)'; m.data(nx632+17)';...
        m.data(nx632+18)'; m.data(nx632+19)'],[],1),'uint32');
    % check for duplicate sqmi logs:
    M = NaN.*zeros(length(obs632),max(obs632));
    for k = 1:double(max(obs632));
        x1 = find(obs632 >= k);
        M(x1,k) = m.data(nx632(x1)+H+4+(k-1)*40);
    end
    minvec = NaN.*zeros(double(max(obs632))-1,1);
    for k = 2:double(max(obs632))
        minvec(k-1) = min(abs(M(:,k)-M(:,k-1)));
    end
%     disp(['min(minvec) is ' num2str(min(minvec))])
    sow_sqm = double(ms632)./1000;
    % get number of accumulation values from first record (ASSUMING ALL ARE
    % THE SAME)
    num_accum = nanmedian(double(typecast(reshape([m.data(nx632+H+12)'; m.data(nx632+H+12+1)';...
        m.data(nx632+H+12+2)'; m.data(nx632+H+12+3)'],[],1),'uint32')));
    
    PRN1sqm = NaN.*zeros(num_accum,length(sow_sqm));
    for k = double(sort(unique(obs632),'descend'))' % do this backwards so MATLAB has a good chance at only having to allocate memory once
        nx = find(obs632==k);
        for kk = k:-1:1
            sigchn = double(typecast(reshape([m.data(nx632(nx)+H+8+4*(3+num_accum)*(kk-1))'; m.data(nx632(nx)+H+8+4*(3+num_accum)*(kk-1)+1)';...
                m.data(nx632(nx)+H+8+4*(3+num_accum)*(kk-1)+2)'; m.data(nx632(nx)+H+8+4*(3+num_accum)*(kk-1)+3)'],[],1),'uint32'));
            % THIS IS FOR TEST G-II ONLY with PRN 1 on Channel 0 (SigChan 0):
            nx1 = nx(sigchn==0); % PRN 1 observations on SV Chan 0 (Sig Chan 0)
            % THIS IS FOR TEST G-II ONLY with PRN 3 on Channel 1 (SigChan 2):
%             nx1 = nx(sigchn==2); % PRN 1 observations on SV Chan 4 (Sig Chan 8)
            % THIS IS FOR TEST G-II ONLY with PRN 1 on Channel 4 (SigChan 8):
%             nx1 = nx(sigchn==8); % PRN 1 observations on SV Chan 4 (Sig Chan 8)
            % THIS IS FOR TEST G-II ONLY with PRN 1 on Channel 10 (SigChan 24):
%             nx1 = nx(sigchn==24); % PRN 7 observations on SV Chan 10 (Sig Chan 24)
            % THIS IS FOR G-II with PRN 135 on Channel 14 (SigChan 32):
%             nx1 = nx(sigchn==32); % PRN 135 observations on SV Chan 14 (Sig Chan 32)
            % THIS IS FOR G-II with PRN 138 on Channel 15 (SigChan 33):
%             nx1 = nx(sigchn==33); % PRN 138 observations on SV Chan 15 (Sig Chan 33)
            % THIS IS FOR GUST with PRN 138 on Channel 14 (SigChan 28):
%             nx1 = nx(sigchn==28); % PRN 138 observations on SV Chan 14 (Sig Chan 28)
            if ~isempty(nx1)
                for n = 1:num_accum
                PRN1sqm(n,nx1) = typecast(reshape([m.data(nx632(nx1)+H+(n*4+12)+4*(3+num_accum)*(kk-1))'; m.data(nx632(nx1)+H+(n*4+12)+4*(3+num_accum)*(kk-1)+1)';...
                    m.data(nx632(nx1)+H+(n*4+12)+4*(3+num_accum)*(kk-1)+2)'; m.data(nx632(nx1)+H+(n*4+12)+4*(3+num_accum)*(kk-1)+3)'],[],1),'int32');
                end
            end
        end
    end
    %compute PR error using correlator measurements
    sqme0p1 = PRN1sqm(1,:)./PRN1sqm(5,:);% .*0.9./0.899545728074703;
    sqme0p075 = PRN1sqm(2,:)./PRN1sqm(5,:);% .*0.925./0.924692012505139;
    sqme0p05 = PRN1sqm(3,:)./PRN1sqm(5,:);% .*0.95./0.953127499351574;
    sqme0p025 = PRN1sqm(4,:)./PRN1sqm(5,:);% .*0.975./0.978947161650227;
    sqml0p025 = PRN1sqm(6,:)./PRN1sqm(5,:);% .*0.975./0.97869403557989;
    sqml0p05 = PRN1sqm(7,:)./PRN1sqm(5,:);% .*0.95./0.953125764781133;
    sqml0p075 = PRN1sqm(8,:)./PRN1sqm(5,:);% .*0.925./0.925175209766234;
    sqml0p1 = PRN1sqm(9,:)./PRN1sqm(5,:);% .*0.9./0.899957077111178;
    s0p025 = 299792458*(0.025575/1.023e6); % 0.025575 chip converted to meters
    s0p075 = 299792458*(0.076725/1.023e6); % 0.076725 chip converted to meters
    s0p1 = 299792458*(0.1023/1.023e6); % 0.1023 chip converted to meters
    s0p05 = 299792458*(0.05115/1.023e6); % 0.05115 chip converted to meters
    pr1err05_1 = (s0p1.*(sqme0p05 - sqml0p05) - s0p05.*(sqme0p1 - sqml0p1))./((sqme0p05 - sqme0p1) + (sqml0p05 - sqml0p1));
    pr1err025_05 = (s0p05.*(sqme0p025 - sqml0p025) - s0p025.*(sqme0p05 - sqml0p05))./((sqme0p025 - sqme0p05) + (sqml0p025 - sqml0p05));
    pr1err05_075 = (s0p075.*(sqme0p05 - sqml0p05) - s0p05.*(sqme0p075 - sqml0p075))./((sqme0p05 - sqme0p075) + (sqml0p05 - sqml0p075));
    pr1err075_1 = (s0p1.*(sqme0p075 - sqml0p075) - s0p075.*(sqme0p1 - sqml0p1))./((sqme0p075 - sqme0p1) + (sqml0p075 - sqml0p1));
end

%%
% get RAWGPSSUBFRAMEWP log index, ID:570, or uint8([58 2])
nxrx = find(m.data(nxrec+4)==58 & m.data(nxrec+5)==2);
if ~isempty(nxrx)
    fprintf('RAWGPSSUBFRAMEWP...')
    nx570 = nxrec(nxrx);
    par570 = typecast(reshape([m.data(nx570+H+8)'; m.data(nx570+H+9)';...
        m.data(nx570+H+10)'; m.data(nx570+H+11)'],[],1),'uint32');
    nx570 = nx570(par570==0); % only taking records with passed parity in subframe for now
    par570 = typecast(reshape([m.data(nx570+H+8)'; m.data(nx570+H+9)';...
        m.data(nx570+H+10)'; m.data(nx570+H+11)'],[],1),'uint32');
    chan570 = typecast(reshape([m.data(nx570+H)'; m.data(nx570+H+1)';...
        m.data(nx570+H+2)'; m.data(nx570+H+3)'],[],1),'uint32');
    msgln570 = typecast(reshape([m.data(nx570+8)'; m.data(nx570+9)'],[],1),'uint16');
    ms570 = typecast(reshape([m.data(nx570+16)'; m.data(nx570+17)';...
        m.data(nx570+18)'; m.data(nx570+19)'],[],1),'uint32');
    sow570 = double(ms570)./1000;
    prn570 = typecast(reshape([m.data(nx570+H+4)'; m.data(nx570+H+5)';...
        m.data(nx570+H+6)'; m.data(nx570+H+7)'],[],1),'uint32');
    pre570 = m.data(nx570+H+12)'; % should all == 139
    towcount = bitor(bitor(bitshift(bitand(uint32(m.data(nx570+H+15)'),3),15),...
        bitshift(uint32(m.data(nx570+H+16)'),7)),bitshift(uint32(m.data(nx570+H+17)'),-1));
    nxflip = find(bitxor(towcount,uint32(131071))-uint32(sow570'./6+1)==0);
    towcount(nxflip) = bitxor(towcount(nxflip),uint32(131071));
    
end

%%
% get FRONTENDDATA log index, ID: 1306, or unit8([26 5])
nx1306 = nxrec(m.data(nxrec+4)==26 & m.data(nxrec+5)==5);
got0fedata = 0;
if ~isempty(nx1306) && parse_frontenddata
    fprintf('FRONTENDDATA...')
    msgln1306 = typecast(reshape([m.data(nx1306+8)'; m.data(nx1306+9)'],[],1),'uint16');
    wk1306 = typecast(reshape([m.data(nx1306+14)'; m.data(nx1306+15)'],[],1),'uint16');
    ms1306 = typecast(reshape([m.data(nx1306+16)'; m.data(nx1306+17)'; m.data(nx1306+18)'; m.data(nx1306+19)'],[],1),'uint32');
    sow1306 = double(ms1306)./1000; % timestamps
    obs1306 = typecast(reshape([m.data(nx1306+H)'; m.data(nx1306+H+1)';...
        m.data(nx1306+H+2)'; m.data(nx1306+H+3)'],[],1),'uint32');
    nx1306 = nx1306(obs1306==median(obs1306(~isnan(obs1306)))); % should all be the same, but just to make sure.
    %fe_timestamp_L1 = NaN.*zeros(size(nx1306));
    %fe_timestamp_L2 = NaN.*zeros(size(nx1306)); % always zero
    %fe_timestamp_L5 = NaN.*zeros(size(nx1306));
    fe_calibrated_L1 = NaN.*zeros(size(nx1306));
    fe_calibrated_L2 = NaN.*zeros(size(nx1306));
    fe_1_calibrated_L1 = NaN.*zeros(size(nx1306));
    fe_1_calibrated_L2 = NaN.*zeros(size(nx1306));
    fe_adjustmode_L1 = NaN.*zeros(size(nx1306));
    fe_adjustmode_L2 = NaN.*zeros(size(nx1306));
    fe_1_adjustmode_L1 = NaN.*zeros(size(nx1306));
    fe_1_adjustmode_L2 = NaN.*zeros(size(nx1306)); 
    fe_adjustrate_L1 = NaN.*zeros(size(nx1306));
    fe_adjustrate_L2 = NaN.*zeros(size(nx1306));
    fe_1_adjustrate_L1 = NaN.*zeros(size(nx1306));
    fe_1_adjustrate_L2 = NaN.*zeros(size(nx1306));
    fe_pulsewidth_L1 = NaN.*zeros(size(nx1306));
    fe_pulsewidth_L2 = NaN.*zeros(size(nx1306));
    fe_1_pulsewidth_L1 = NaN.*zeros(size(nx1306));
    fe_1_pulsewidth_L2 = NaN.*zeros(size(nx1306));
    %fe_modulus_L1 = NaN.*zeros(size(nx1306));
    %fe_modulus_L2 = NaN.*zeros(size(nx1306)); % always 8000
    %fe_modulus_L5 = NaN.*zeros(size(nx1306));
    fe_bitrange_L1 = NaN.*zeros(size(nx1306));
    fe_bitrange_L2 = NaN.*zeros(size(nx1306));
    fe_1_bitrange_L1 = NaN.*zeros(size(nx1306));
    fe_1_bitrange_L2 = NaN.*zeros(size(nx1306));
    fe_dcoffset_L1 = NaN.*zeros(size(nx1306));
    fe_dcoffset_L2 = NaN.*zeros(size(nx1306));
    fe_1_dcoffset_L1 = NaN.*zeros(size(nx1306));
    fe_1_dcoffset_L2 = NaN.*zeros(size(nx1306));
    fe_pdferror_L1 = NaN.*zeros(size(nx1306));
    fe_pdferror_L2 = NaN.*zeros(size(nx1306));
    fe_1_pdferror_L1 = NaN.*zeros(size(nx1306));
    fe_1_pdferror_L2 = NaN.*zeros(size(nx1306));
    fe_pdf_L1 = NaN.*zeros(6,length(nx1306));
    fe_pdf_L2 = NaN.*zeros(6,length(nx1306));
    fe_1_pdf_L1 = NaN.*zeros(6,length(nx1306));
    fe_1_pdf_L2 = NaN.*zeros(6,length(nx1306));
    for k = 1:double(median(obs1306(~isnan(obs1306))))
        rftype = typecast(reshape([m.data(nx1306+H+4+(k-1)*96)'; m.data(nx1306+H+5+(k-1)*96)'; m.data(nx1306+H+6+(k-1)*96)'; m.data(nx1306+H+7+(k-1)*96)'],[],1),'uint32');      
        nx = find(rftype == 0); % L1 agc data
        if ~isempty(nx) && ~got0fedata
            %fe_timestamp_L1(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+8+(k-1)*96)'; m.data(nx1306(nx)+H+9+(k-1)*96)'; m.data(nx1306(nx)+H+10+(k-1)*96)'; m.data(nx1306(nx)+H+11+(k-1)*96)'],[],1),'uint32'));
            fe_calibrated_L1(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+12+(k-1)*96)'; m.data(nx1306(nx)+H+13+(k-1)*96)'; m.data(nx1306(nx)+H+14+(k-1)*96)'; m.data(nx1306(nx)+H+15+(k-1)*96)'],[],1),'uint32'));
            fe_adjustmode_L1(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+16+(k-1)*96)'; m.data(nx1306(nx)+H+17+(k-1)*96)'; m.data(nx1306(nx)+H+18+(k-1)*96)'; m.data(nx1306(nx)+H+19+(k-1)*96)'],[],1),'uint32'));
            fe_adjustrate_L1(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+20+(k-1)*96)'; m.data(nx1306(nx)+H+21+(k-1)*96)'; m.data(nx1306(nx)+H+22+(k-1)*96)'; m.data(nx1306(nx)+H+23+(k-1)*96)'],[],1),'uint32'));
            fe_pulsewidth_L1(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+24+(k-1)*96)'; m.data(nx1306(nx)+H+25+(k-1)*96)'; m.data(nx1306(nx)+H+26+(k-1)*96)'; m.data(nx1306(nx)+H+27+(k-1)*96)'],[],1),'uint32'));
            %fe_modulus_L1(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+28+(k-1)*96)'; m.data(nx1306(nx)+H+29+(k-1)*96)'; m.data(nx1306(nx)+H+30+(k-1)*96)'; m.data(nx1306(nx)+H+31+(k-1)*96)'],[],1),'uint32'));
            fe_bitrange_L1(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+32+(k-1)*96)'; m.data(nx1306(nx)+H+33+(k-1)*96)'; m.data(nx1306(nx)+H+34+(k-1)*96)'; m.data(nx1306(nx)+H+35+(k-1)*96)'],[],1),'uint32'));
            fe_dcoffset_L1(nx) = typecast(reshape([m.data(nx1306(nx)+H+36+(k-1)*96)'; m.data(nx1306(nx)+H+37+(k-1)*96)';...
                m.data(nx1306(nx)+H+38+(k-1)*96)'; m.data(nx1306(nx)+H+39+(k-1)*96)';...
                m.data(nx1306(nx)+H+40+(k-1)*96)'; m.data(nx1306(nx)+H+41+(k-1)*96)';...
                m.data(nx1306(nx)+H+42+(k-1)*96)'; m.data(nx1306(nx)+H+43+(k-1)*96)'],[],1),'double'); 
            fe_pdferror_L1(nx) = typecast(reshape([m.data(nx1306(nx)+H+44+(k-1)*96)'; m.data(nx1306(nx)+H+45+(k-1)*96)';...
                m.data(nx1306(nx)+H+46+(k-1)*96)'; m.data(nx1306(nx)+H+47+(k-1)*96)';...
                m.data(nx1306(nx)+H+48+(k-1)*96)'; m.data(nx1306(nx)+H+49+(k-1)*96)';...
                m.data(nx1306(nx)+H+50+(k-1)*96)'; m.data(nx1306(nx)+H+51+(k-1)*96)'],[],1),'double'); 
            for kk = 6:-1:1
                fe_pdf_L1(kk,nx) = typecast(reshape([m.data(nx1306(nx)+H+kk*8+44+(k-1)*96)'; m.data(nx1306(nx)+H+kk*8+45+(k-1)*96)';...
                    m.data(nx1306(nx)+H+kk*8+46+(k-1)*96)'; m.data(nx1306(nx)+H+kk*8+47+(k-1)*96)';...
                    m.data(nx1306(nx)+H+kk*8+48+(k-1)*96)'; m.data(nx1306(nx)+H+kk*8+49+(k-1)*96)';...
                    m.data(nx1306(nx)+H+kk*8+50+(k-1)*96)'; m.data(nx1306(nx)+H+kk*8+51+(k-1)*96)'],[],1),'double'); 
            end
            got0fedata=1;
        end
        nx = find(rftype == 2); % L2 agc data
        if ~isempty(nx)
            %fe_timestamp_L2(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+8+(k-1)*96)'; m.data(nx1306(nx)+H+9+(k-1)*96)'; m.data(nx1306(nx)+H+10+(k-1)*96)'; m.data(nx1306(nx)+H+11+(k-1)*96)'],[],1),'uint32'));
            fe_calibrated_L2(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+12+(k-1)*96)'; m.data(nx1306(nx)+H+13+(k-1)*96)'; m.data(nx1306(nx)+H+14+(k-1)*96)'; m.data(nx1306(nx)+H+15+(k-1)*96)'],[],1),'uint32'));
            fe_adjustmode_L2(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+16+(k-1)*96)'; m.data(nx1306(nx)+H+17+(k-1)*96)'; m.data(nx1306(nx)+H+18+(k-1)*96)'; m.data(nx1306(nx)+H+19+(k-1)*96)'],[],1),'uint32'));
            fe_adjustrate_L2(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+20+(k-1)*96)'; m.data(nx1306(nx)+H+21+(k-1)*96)'; m.data(nx1306(nx)+H+22+(k-1)*96)'; m.data(nx1306(nx)+H+23+(k-1)*96)'],[],1),'uint32'));
            fe_pulsewidth_L2(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+24+(k-1)*96)'; m.data(nx1306(nx)+H+25+(k-1)*96)'; m.data(nx1306(nx)+H+26+(k-1)*96)'; m.data(nx1306(nx)+H+27+(k-1)*96)'],[],1),'uint32'));
            %fe_modulus_L2(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+28+(k-1)*96)'; m.data(nx1306(nx)+H+29+(k-1)*96)'; m.data(nx1306(nx)+H+30+(k-1)*96)'; m.data(nx1306(nx)+H+31+(k-1)*96)'],[],1),'uint32'));
            fe_bitrange_L2(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+32+(k-1)*96)'; m.data(nx1306(nx)+H+33+(k-1)*96)'; m.data(nx1306(nx)+H+34+(k-1)*96)'; m.data(nx1306(nx)+H+35+(k-1)*96)'],[],1),'uint32'));
            fe_dcoffset_L2(nx) = typecast(reshape([m.data(nx1306(nx)+H+36+(k-1)*96)'; m.data(nx1306(nx)+H+37+(k-1)*96)';...
                m.data(nx1306(nx)+H+38+(k-1)*96)'; m.data(nx1306(nx)+H+39+(k-1)*96)';...
                m.data(nx1306(nx)+H+40+(k-1)*96)'; m.data(nx1306(nx)+H+41+(k-1)*96)';...
                m.data(nx1306(nx)+H+42+(k-1)*96)'; m.data(nx1306(nx)+H+43+(k-1)*96)'],[],1),'double'); 
            fe_pdferror_L2(nx) = typecast(reshape([m.data(nx1306(nx)+H+44+(k-1)*96)'; m.data(nx1306(nx)+H+45+(k-1)*96)';...
                m.data(nx1306(nx)+H+46+(k-1)*96)'; m.data(nx1306(nx)+H+47+(k-1)*96)';...
                m.data(nx1306(nx)+H+48+(k-1)*96)'; m.data(nx1306(nx)+H+49+(k-1)*96)';...
                m.data(nx1306(nx)+H+50+(k-1)*96)'; m.data(nx1306(nx)+H+51+(k-1)*96)'],[],1),'double'); 
            for kk = 6:-1:1
                fe_pdf_L2(kk,nx) = typecast(reshape([m.data(nx1306(nx)+H+kk*8+44+(k-1)*96)'; m.data(nx1306(nx)+H+kk*8+45+(k-1)*96)';...
                    m.data(nx1306(nx)+H+kk*8+46+(k-1)*96)'; m.data(nx1306(nx)+H+kk*8+47+(k-1)*96)';...
                    m.data(nx1306(nx)+H+kk*8+48+(k-1)*96)'; m.data(nx1306(nx)+H+kk*8+49+(k-1)*96)';...
                    m.data(nx1306(nx)+H+kk*8+50+(k-1)*96)'; m.data(nx1306(nx)+H+kk*8+51+(k-1)*96)'],[],1),'double'); 
            end
        end
        nx = find(rftype == 3); % ANT2 L1 agc data
        if ~isempty(nx)
            %fe_timestamp_L5(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+8+(k-1)*96)'; m.data(nx1306(nx)+H+9+(k-1)*96)'; m.data(nx1306(nx)+H+10+(k-1)*96)'; m.data(nx1306(nx)+H+11+(k-1)*96)'],[],1),'uint32'));
            fe_1_calibrated_L1(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+12+(k-1)*96)'; m.data(nx1306(nx)+H+13+(k-1)*96)'; m.data(nx1306(nx)+H+14+(k-1)*96)'; m.data(nx1306(nx)+H+15+(k-1)*96)'],[],1),'uint32'));
            fe_1_adjustmode_L1(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+16+(k-1)*96)'; m.data(nx1306(nx)+H+17+(k-1)*96)'; m.data(nx1306(nx)+H+18+(k-1)*96)'; m.data(nx1306(nx)+H+19+(k-1)*96)'],[],1),'uint32'));
            fe_1_adjustrate_L1(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+20+(k-1)*96)'; m.data(nx1306(nx)+H+21+(k-1)*96)'; m.data(nx1306(nx)+H+22+(k-1)*96)'; m.data(nx1306(nx)+H+23+(k-1)*96)'],[],1),'uint32'));
            fe_1_pulsewidth_L1(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+24+(k-1)*96)'; m.data(nx1306(nx)+H+25+(k-1)*96)'; m.data(nx1306(nx)+H+26+(k-1)*96)'; m.data(nx1306(nx)+H+27+(k-1)*96)'],[],1),'uint32'));
            %fe_modulus_L5(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+28+(k-1)*96)'; m.data(nx1306(nx)+H+29+(k-1)*96)'; m.data(nx1306(nx)+H+30+(k-1)*96)'; m.data(nx1306(nx)+H+31+(k-1)*96)'],[],1),'uint32'));
            fe_1_bitrange_L1(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+32+(k-1)*96)'; m.data(nx1306(nx)+H+33+(k-1)*96)'; m.data(nx1306(nx)+H+34+(k-1)*96)'; m.data(nx1306(nx)+H+35+(k-1)*96)'],[],1),'uint32'));
            fe_1_dcoffset_L1(nx) = typecast(reshape([m.data(nx1306(nx)+H+36+(k-1)*96)'; m.data(nx1306(nx)+H+37+(k-1)*96)';...
                m.data(nx1306(nx)+H+38+(k-1)*96)'; m.data(nx1306(nx)+H+39+(k-1)*96)';...
                m.data(nx1306(nx)+H+40+(k-1)*96)'; m.data(nx1306(nx)+H+41+(k-1)*96)';...
                m.data(nx1306(nx)+H+42+(k-1)*96)'; m.data(nx1306(nx)+H+43+(k-1)*96)'],[],1),'double'); 
            fe_1_pdferror_L1(nx) = typecast(reshape([m.data(nx1306(nx)+H+44+(k-1)*96)'; m.data(nx1306(nx)+H+45+(k-1)*96)';...
                m.data(nx1306(nx)+H+46+(k-1)*96)'; m.data(nx1306(nx)+H+47+(k-1)*96)';...
                m.data(nx1306(nx)+H+48+(k-1)*96)'; m.data(nx1306(nx)+H+49+(k-1)*96)';...
                m.data(nx1306(nx)+H+50+(k-1)*96)'; m.data(nx1306(nx)+H+51+(k-1)*96)'],[],1),'double'); 
            for kk = 6:-1:1
                fe_1_pdf_L1(kk,nx) = typecast(reshape([m.data(nx1306(nx)+H+kk*8+44+(k-1)*96)'; m.data(nx1306(nx)+H+kk*8+45+(k-1)*96)';...
                    m.data(nx1306(nx)+H+kk*8+46+(k-1)*96)'; m.data(nx1306(nx)+H+kk*8+47+(k-1)*96)';...
                    m.data(nx1306(nx)+H+kk*8+48+(k-1)*96)'; m.data(nx1306(nx)+H+kk*8+49+(k-1)*96)';...
                    m.data(nx1306(nx)+H+kk*8+50+(k-1)*96)'; m.data(nx1306(nx)+H+kk*8+51+(k-1)*96)'],[],1),'double'); 
            end
        end
        nx = find(rftype == 4); % ANT2 L2 agc data
        if ~isempty(nx)
            %fe_timestamp_L5(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+8+(k-1)*96)'; m.data(nx1306(nx)+H+9+(k-1)*96)'; m.data(nx1306(nx)+H+10+(k-1)*96)'; m.data(nx1306(nx)+H+11+(k-1)*96)'],[],1),'uint32'));
            fe_1_calibrated_L2(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+12+(k-1)*96)'; m.data(nx1306(nx)+H+13+(k-1)*96)'; m.data(nx1306(nx)+H+14+(k-1)*96)'; m.data(nx1306(nx)+H+15+(k-1)*96)'],[],1),'uint32'));
            fe_1_adjustmode_L2(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+16+(k-1)*96)'; m.data(nx1306(nx)+H+17+(k-1)*96)'; m.data(nx1306(nx)+H+18+(k-1)*96)'; m.data(nx1306(nx)+H+19+(k-1)*96)'],[],1),'uint32'));
            fe_1_adjustrate_L2(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+20+(k-1)*96)'; m.data(nx1306(nx)+H+21+(k-1)*96)'; m.data(nx1306(nx)+H+22+(k-1)*96)'; m.data(nx1306(nx)+H+23+(k-1)*96)'],[],1),'uint32'));
            fe_1_pulsewidth_L2(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+24+(k-1)*96)'; m.data(nx1306(nx)+H+25+(k-1)*96)'; m.data(nx1306(nx)+H+26+(k-1)*96)'; m.data(nx1306(nx)+H+27+(k-1)*96)'],[],1),'uint32'));
            %fe_modulus_L5(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+28+(k-1)*96)'; m.data(nx1306(nx)+H+29+(k-1)*96)'; m.data(nx1306(nx)+H+30+(k-1)*96)'; m.data(nx1306(nx)+H+31+(k-1)*96)'],[],1),'uint32'));
            fe_1_bitrange_L2(nx) = double(typecast(reshape([m.data(nx1306(nx)+H+32+(k-1)*96)'; m.data(nx1306(nx)+H+33+(k-1)*96)'; m.data(nx1306(nx)+H+34+(k-1)*96)'; m.data(nx1306(nx)+H+35+(k-1)*96)'],[],1),'uint32'));
            fe_1_dcoffset_L2(nx) = typecast(reshape([m.data(nx1306(nx)+H+36+(k-1)*96)'; m.data(nx1306(nx)+H+37+(k-1)*96)';...
                m.data(nx1306(nx)+H+38+(k-1)*96)'; m.data(nx1306(nx)+H+39+(k-1)*96)';...
                m.data(nx1306(nx)+H+40+(k-1)*96)'; m.data(nx1306(nx)+H+41+(k-1)*96)';...
                m.data(nx1306(nx)+H+42+(k-1)*96)'; m.data(nx1306(nx)+H+43+(k-1)*96)'],[],1),'double'); 
            fe_1_pdferror_L2(nx) = typecast(reshape([m.data(nx1306(nx)+H+44+(k-1)*96)'; m.data(nx1306(nx)+H+45+(k-1)*96)';...
                m.data(nx1306(nx)+H+46+(k-1)*96)'; m.data(nx1306(nx)+H+47+(k-1)*96)';...
                m.data(nx1306(nx)+H+48+(k-1)*96)'; m.data(nx1306(nx)+H+49+(k-1)*96)';...
                m.data(nx1306(nx)+H+50+(k-1)*96)'; m.data(nx1306(nx)+H+51+(k-1)*96)'],[],1),'double'); 
            for kk = 6:-1:1
                fe_1_pdf_L2(kk,nx) = typecast(reshape([m.data(nx1306(nx)+H+kk*8+44+(k-1)*96)'; m.data(nx1306(nx)+H+kk*8+45+(k-1)*96)';...
                    m.data(nx1306(nx)+H+kk*8+46+(k-1)*96)'; m.data(nx1306(nx)+H+kk*8+47+(k-1)*96)';...
                    m.data(nx1306(nx)+H+kk*8+48+(k-1)*96)'; m.data(nx1306(nx)+H+kk*8+49+(k-1)*96)';...
                    m.data(nx1306(nx)+H+kk*8+50+(k-1)*96)'; m.data(nx1306(nx)+H+kk*8+51+(k-1)*96)'],[],1),'double'); 
            end
        end
    end
end

%%
% get SATVIS log index, ID: 48, or uint8([48 0])
% nx048 = nxrec(m.data(nxrec+4)==48 & m.data(nxrec+5)==0);
% if ~isempty(nx048)
%     fprintf('SATVIS...')
%     svvalid = typecast(reshape([m.data(nx048+H+0)'; m.data(nx048+H+1)';...
%         m.data(nx048+H+2)'; m.data(nx048+H+3)'],[],1),'uint32');
%     nx048 = nx048(svvalid==uint32(1));
%     numsatdat = double(typecast(reshape([m.data(nx048+H+8)'; m.data(nx048+H+9)';...
%         m.data(nx048+H+10)'; m.data(nx048+H+11)'],[],1),'uint32'));
%     numsats = unique(numsatdat);
%     if length(numsats) > 1
%         disp('numsats for SATVIS changes in this file (need to change script logic)')
%     else
%         sow048 = double(typecast(reshape([m.data(nx048+16)'; m.data(nx048+17)';...
%             m.data(nx048+18)'; m.data(nx048+19)'],[],1),'uint32'))./1000;
%         Elsat = NaN.*zeros(32,length(nx048));
%         Azsat = NaN.*zeros(32,length(nx048));
% %         TrueDop = NaN.*zeros(32,length(nx048));
% %         AppDop = NaN.*zeros(32,length(nx048));
%         for k = 1:numsats
%             prndx = typecast(reshape([m.data(nx048+H+(k-1)*40+12)'; m.data(nx048+H+(k-1)*40+13)'],[],1),'uint16');
%             kprns = unique(prndx);
%             for kk = 1:length(kprns)
%                 nx1 = find(prndx==kprns(kk));
%                 Elsat(kprns(kk),nx1) = typecast(reshape([m.data(nx048(nx1)+H+20+40*(k-1))'; m.data(nx048(nx1)+H+21+40*(k-1))';...
%                     m.data(nx048(nx1)+H+22+40*(k-1))'; m.data(nx048(nx1)+H+23+40*(k-1))';...
%                     m.data(nx048(nx1)+H+24+40*(k-1))'; m.data(nx048(nx1)+H+25+40*(k-1))';...
%                     m.data(nx048(nx1)+H+26+40*(k-1))'; m.data(nx048(nx1)+H+27+40*(k-1))'],[],1),'double');
%                 Azsat(kprns(kk),nx1) = typecast(reshape([m.data(nx048(nx1)+H+28+40*(k-1))'; m.data(nx048(nx1)+H+29+40*(k-1))';...
%                     m.data(nx048(nx1)+H+30+40*(k-1))'; m.data(nx048(nx1)+H+31+40*(k-1))';...
%                     m.data(nx048(nx1)+H+32+40*(k-1))'; m.data(nx048(nx1)+H+33+40*(k-1))';...
%                     m.data(nx048(nx1)+H+34+40*(k-1))'; m.data(nx048(nx1)+H+35+40*(k-1))'],[],1),'double');
% %                 TrueDop(kprns(kk),nx1) = typecast(reshape([m.data(nx048(nx1)+H+36+40*(k-1))'; m.data(nx048(nx1)+H+37+40*(k-1))';...
% %                     m.data(nx048(nx1)+H+38+40*(k-1))'; m.data(nx048(nx1)+H+39+40*(k-1))';...
% %                     m.data(nx048(nx1)+H+40+40*(k-1))'; m.data(nx048(nx1)+H+41+40*(k-1))';...
% %                     m.data(nx048(nx1)+H+42+40*(k-1))'; m.data(nx048(nx1)+H+43+40*(k-1))'],[],1),'double');
% %                 AppDop(kprns(kk),nx1) = typecast(reshape([m.data(nx048(nx1)+H+44+40*(k-1))'; m.data(nx048(nx1)+H+45+40*(k-1))';...
% %                     m.data(nx048(nx1)+H+46+40*(k-1))'; m.data(nx048(nx1)+H+47+40*(k-1))';...
% %                     m.data(nx048(nx1)+H+48+40*(k-1))'; m.data(nx048(nx1)+H+49+40*(k-1))';...
% %                     m.data(nx048(nx1)+H+50+40*(k-1))'; m.data(nx048(nx1)+H+51+40*(k-1))'],[],1),'double');
%             end
%         end
%     end
%     El_rge1 = NaN.*PRN_rge1;
%     Az_rge1 = NaN.*PRN_rge1;
% %     for k = 1:32
% %         [nxr,nxc] = find(PRN_rge1==k);
% %         nxx = find(PRN_rge1==k);
% %         sowprnk = sow_rge(nxc);
% %         [sowi,ixrge,ixsat] = intersect(sowprnk,sow048);
% %         El_rge1(nxx(ixrge)) = Elsat(k,ixsat);
% %         Az_rge1(nxx(ixrge)) = Azsat(k,ixsat);
% %     end
%     %%
%     figure
%     clrmtx = jet(20);
%     nxx = find(CNo1 <= 36);
%     hp = polar(deg2rad(90-Az_rge1(nxx)'),(90-El_rge1(nxx))'./15,'.');
%     set(hp,'Color',clrmtx(1,:))
%     hold on
%     [x,y]=pol2cart(deg2rad(0:360),((90-5)./15).*ones(size(0:360)));
%     hc5 = plot(x,y,'Color','k');
%     for CNok = 36:55
%         nxx = find(CNo1 > CNok-1 & CNo1 <= CNok);
%         hp = polar(deg2rad(90-Az_rge1(nxx)'),(90-El_rge1(nxx))'./15,'.');
%         set(hp,'Color',clrmtx(CNok-35,:))
%     end
%     nxx = find(CNo1 > 55);
%     hp = polar(deg2rad(90-Az_rge1(nxx)'),(90-El_rge1(nxx))'./15,'.');
%     set(hp,'Color',clrmtx(end,:))
%     set(gcf,'Position',[420 338 560 612],'PaperPosition',[0.5 0.5 6 6.*612./560]);
%     set(gca,'Position',[0.1 0.11 0.8 0.815]);
%     colormap(jet(20));
%     hcb = colorbar('Location','southoutside');
%     set(hcb,'Position',[0.125 0.05 0.75 0.04])
%     set(hcb,'XTick',1+([36 37 40 45 50 55 56]-36))
%     set(hcb,'XTickLabel',{'<', '37','40','45','50','55', '>'})
%     title('GPS L1 C/A C/No by Elevation and Azimuth')
% %%    
%     figure
%     plot(El_rge1',CNo1')
%     set(gca,'YLim',[20 60])
%     grid on
%     xlabel('elevation angle (deg)')
%     ylabel('C/No (dB-Hz)')
%     set(gca,'Position',[0.08 0.11 0.875 0.815])
%     fpos = get(gcf,'Position');
%     set(gcf,'Position',[fpos(1) fpos(2) 560 373])
%     set(gcf,'PaperPosition',[.5 .5 6 4]);
%     title('GPS L1 C/A C/No by Elevation Angle')
% end

% %%
% % get SATVIS2 log index, ID: 1043, or uint8([19 4])
% nx1043 = nxrec(m.data(nxrec+4)==19 & m.data(nxrec+5)==4);
% if ~isempty(nx1043)
%     fprintf('SATVIS2 Galileo...')
%     svsys = typecast(reshape([m.data(nx1043+H+0)'; m.data(nx1043+H+1)';...
%         m.data(nx1043+H+2)'; m.data(nx1043+H+3)'],[],1),'uint32');
%     svvalid = typecast(reshape([m.data(nx1043+H+4)'; m.data(nx1043+H+5)';...
%         m.data(nx1043+H+6)'; m.data(nx1043+H+7)'],[],1),'uint32');
%     nx1043 = nx1043(svvalid==uint32(1) & svsys==uint32(5));
%     numsatdat = double(typecast(reshape([m.data(nx1043+H+12)'; m.data(nx1043+H+13)';...
%         m.data(nx1043+H+14)'; m.data(nx1043+H+15)'],[],1),'uint32'));
%     numsats = unique(numsatdat);
%     if length(numsats) > 1
%         disp('numsats for SATVIS2 Galileo changes in this file (need to change script logic)')
%     else
%         sow1043 = double(typecast(reshape([m.data(nx1043+16)'; m.data(nx1043+17)';...
%             m.data(nx1043+18)'; m.data(nx1043+19)'],[],1),'uint32'))./1000;
%         Elsatg = NaN.*zeros(32,length(nx1043));
%         Azsatg = NaN.*zeros(32,length(nx1043));
% %         TrueDop = NaN.*zeros(32,length(nx1043));
% %         AppDop = NaN.*zeros(32,length(nx1043));
%         for k = 1:numsats
%             prndx = typecast(reshape([m.data(nx1043+H+(k-1)*40+16)'; m.data(nx1043+H+(k-1)*40+17)';...
%                 m.data(nx1043+H+(k-1)*40+18)'; m.data(nx1043+H+(k-1)*40+19)'],[],1),'uint32');
%             kprns = unique(prndx);
%             for kk = 1:length(kprns)
%                 nx1 = find(prndx==kprns(kk));
%                 Elsatg(kprns(kk),nx1) = typecast(reshape([m.data(nx1043(nx1)+H+24+40*(k-1))'; m.data(nx1043(nx1)+H+25+40*(k-1))';...
%                     m.data(nx1043(nx1)+H+26+40*(k-1))'; m.data(nx1043(nx1)+H+27+40*(k-1))';...
%                     m.data(nx1043(nx1)+H+28+40*(k-1))'; m.data(nx1043(nx1)+H+29+40*(k-1))';...
%                     m.data(nx1043(nx1)+H+30+40*(k-1))'; m.data(nx1043(nx1)+H+31+40*(k-1))'],[],1),'double');
%                 Azsatg(kprns(kk),nx1) = typecast(reshape([m.data(nx1043(nx1)+H+32+40*(k-1))'; m.data(nx1043(nx1)+H+33+40*(k-1))';...
%                     m.data(nx1043(nx1)+H+34+40*(k-1))'; m.data(nx1043(nx1)+H+35+40*(k-1))';...
%                     m.data(nx1043(nx1)+H+36+40*(k-1))'; m.data(nx1043(nx1)+H+37+40*(k-1))';...
%                     m.data(nx1043(nx1)+H+38+40*(k-1))'; m.data(nx1043(nx1)+H+39+40*(k-1))'],[],1),'double');
% %                 TrueDop(kprns(kk),nx1) = typecast(reshape([m.data(nx1043(nx1)+H+40+40*(k-1))'; m.data(nx1043(nx1)+H+41+40*(k-1))';...
% %                     m.data(nx1043(nx1)+H+38+42*(k-1))'; m.data(nx1043(nx1)+H+43+40*(k-1))';...
% %                     m.data(nx1043(nx1)+H+44+40*(k-1))'; m.data(nx1043(nx1)+H+45+40*(k-1))';...
% %                     m.data(nx1043(nx1)+H+46+40*(k-1))'; m.data(nx1043(nx1)+H+47+40*(k-1))'],[],1),'double');
% %                 AppDop(kprns(kk),nx1) = typecast(reshape([m.data(nx1043(nx1)+H+48+40*(k-1))'; m.data(nx1043(nx1)+H+49+40*(k-1))';...
% %                     m.data(nx1043(nx1)+H+50+40*(k-1))'; m.data(nx1043(nx1)+H+51+40*(k-1))';...
% %                     m.data(nx1043(nx1)+H+52+40*(k-1))'; m.data(nx1043(nx1)+H+53+40*(k-1))';...
% %                     m.data(nx1043(nx1)+H+54+40*(k-1))'; m.data(nx1043(nx1)+H+55+40*(k-1))'],[],1),'double');
%             end
%         end
%     end
%     El_1e = NaN.*PRN1e;
%     Az_1e = NaN.*PRN1e;
%     for k = 1:32
%         [nxr,nxc] = find(PRN1e==k);
%         nxx = find(PRN1e==k);
%         sowprnk = sow_rge(nxc);
%         [sowi,ixrge,ixsat] = intersect(sowprnk,sow1043);
%         El_1e(nxx(ixrge)) = Elsatg(k,ixsat);
%         Az_1e(nxx(ixrge)) = Azsatg(k,ixsat);
%     end
%     %%
%     figure
%     clrmtx = jet(20);
%     nxx = find(CNo1 <= 36);
%     hp = polar(deg2rad(90-Az_1e(nxx)'),(90-El_1e(nxx))'./15,'.');
%     set(hp,'Color',clrmtx(1,:))
%     hold on
%     [x,y]=pol2cart(deg2rad(0:360),((90-5)./15).*ones(size(0:360)));
%     hc5 = plot(x,y,'Color','k');
%     for CNok = 36:55
%         nxx = find(CNo1e > CNok-1 & CNo1e <= CNok);
%         hp = polar(deg2rad(90-Az_1e(nxx)'),(90-El_1e(nxx))'./15,'.');
%         set(hp,'Color',clrmtx(CNok-35,:))
%     end
%     nxx = find(CNo1e > 55);
%     hp = polar(deg2rad(90-Az_1e(nxx)'),(90-El_1e(nxx))'./15,'.');
%     set(hp,'Color',clrmtx(end,:))
%     set(gcf,'Position',[420 338 560 612],'PaperPosition',[0.5 0.5 6 6.*612./560]);
%     set(gca,'Position',[0.1 0.11 0.8 0.815]);
%     colormap(jet(20));
%     hcb = colorbar('Location','southoutside');
%     set(hcb,'Position',[0.125 0.05 0.75 0.04])
% %     set(hcb,'XTick',1+([36 37 40 45 50 55 56]-36))
%     set(hcb,'XTick',(1+([36 37 40 45 50 55 56]-36)-1)./20)
%     set(hcb,'XTickLabel',{'<', '37','40','45','50','55', '>'})
%     title('Galileo L1 E1C C/No by Elevation and Azimuth')
% %%    
%     figure
% %     plot(El_1e',CNo1e')
%     legtxt = {};
%     legln = 0;
%     clrmtx = get(gca,'ColorOrder');
%     clrmtx(size(clrmtx,1),:) = [1 0 0];
%     clrmtx(size(clrmtx,1)+1,:) = [0 1 0];
%     clrmtx(size(clrmtx,1)+1,:) = [0 0 1];
%     clrmtx(size(clrmtx,1)+1,:) = [0 0 0];
%     set(gca,'ColorOrder',clrmtx);
%     hold on
%     for k = 1:32
%         elnx = find(PRN1e==k);
%         if ~isempty(elnx)
%             plot(El_1e(elnx),CNo1e(elnx),'.')
%             legln = legln+1;
%             legtxt{legln} = num2str(k); %#ok<SAGROW>
%         end
%     end
% %     plot(El_1e',CNo1e')
% %     set(gca,'YLim',[20 60])
%     set(gca,'YLim',[34 54])
%     grid on
%     xlabel('elevation angle (deg)')
%     ylabel('C/No (dB-Hz)')
%     set(gca,'Position',[0.08 0.12 0.875 0.815])
%     fpos = get(gcf,'Position');
%     set(gcf,'Position',[fpos(1) fpos(2) 560 373])
%     set(gcf,'PaperPosition',[.5 .5 6 4]);
%     legend(legtxt,'Location','se')
%     title('Galileo L1 E1C C/No by Elevation Angle')
% end
%%
% get SATVIS log index, ID: 48, or uint8([48 0])
nxrx = find(m.data(nxrec+4)==48 & m.data(nxrec+5)==0);
if ~isempty(nxrx) && parse_satvisb
    fprintf('SATVIS...')
    nx048 = nxrec(nxrx);
    msgln048 = typecast(reshape([m.data(nx048+8)'; m.data(nx048+9)'],[],1),'uint16');
    svvalid = typecast(reshape([m.data(nx048+H+0)'; m.data(nx048+H+1)';...
                                m.data(nx048+H+2)'; m.data(nx048+H+3)'],[],1),'uint32');
    almused = typecast(reshape([m.data(nx048+H+4)'; m.data(nx048+H+5)';...
                                m.data(nx048+H+6)'; m.data(nx048+H+7)'],[],1),'uint32');
    sats048 = typecast(reshape([m.data(nx048+H+8)'; m.data(nx048+H+9)';...
                                m.data(nx048+H+10)'; m.data(nx048+H+11)'],[],1),'uint32');
    idl048 = double(m.data(nx048+12)')./2;
    wk048 = typecast(reshape([m.data(nx048+14)'; m.data(nx048+15)'],[],1),'uint16');
    ms048 = typecast(reshape([m.data(nx048+16)'; m.data(nx048+17)';...
        m.data(nx048+18)'; m.data(nx048+19)'],[],1),'uint32');
    rxstat048 = typecast(reshape([m.data(nx048+20)'; m.data(nx048+21)';...
        m.data(nx048+22)'; m.data(nx048+23)'],[],1),'uint32');
    % the following initialization is to try to avoid resizing matrices:
    [maxobs,maxnx] = max(sats048);

    sow_sats = double(ms048)./1000;

    prn_idx=NaN.*zeros(1,length(nx048));
    PRN_sats=NaN.*zeros(32,length(nx048));
    health=NaN.*zeros(32,length(nx048));
    az=NaN.*zeros(32,length(nx048));
    elev=NaN.*zeros(32,length(nx048));
    true_dop=NaN.*zeros(32,length(nx048));
    app_dop=NaN.*zeros(32,length(nx048));

    for k = double(sort(unique(sats048),'descend'))' % do this backwards so MATLAB has a good chance at only having to allocate memory once
        for kk = k:-1:1
            prn_idx = typecast(reshape([m.data(nx048+H+12+40*(kk-1))'; m.data(nx048+H+13+40*(kk-1))'],[],1),'uint16'); 
            prn32 =  prn_idx > 32; 
            prn_idx(prn32) = 0; % Remove non-valid PRNs. speeds up code
            uni_idx = double(sort(unique(prn_idx),'descend'))';
            for idx = uni_idx
                if idx==0
                    break;
                end
                nx = prn_idx==idx & svvalid==1;
                    PRN_sats(idx,nx) = typecast(reshape([m.data(nx048(nx)+H+12+40*(kk-1))'; m.data(nx048(nx)+H+13+40*(kk-1))'],[],1),'uint16');
                    health(idx,nx)   = typecast(reshape([m.data(nx048(nx)+H+16+40*(kk-1))'; m.data(nx048(nx)+H+17+40*(kk-1))';...
                                                         m.data(nx048(nx)+H+18+40*(kk-1))'; m.data(nx048(nx)+H+19+40*(kk-1))'],[],1),'uint32');
                    elev(idx,nx)       = typecast(reshape([m.data(nx048(nx)+H+20+40*(kk-1))'; m.data(nx048(nx)+H+21+40*(kk-1))';...
                                                         m.data(nx048(nx)+H+22+40*(kk-1))'; m.data(nx048(nx)+H+23+40*(kk-1))';...
                                                         m.data(nx048(nx)+H+24+40*(kk-1))'; m.data(nx048(nx)+H+25+40*(kk-1))';...
                                                         m.data(nx048(nx)+H+26+40*(kk-1))'; m.data(nx048(nx)+H+27+40*(kk-1))'],[],1),'double');
                    az(idx,nx)     = typecast(reshape([m.data(nx048(nx)+H+28+40*(kk-1))'; m.data(nx048(nx)+H+29+40*(kk-1))';...
                                                         m.data(nx048(nx)+H+30+40*(kk-1))'; m.data(nx048(nx)+H+31+40*(kk-1))';...
                                                         m.data(nx048(nx)+H+32+40*(kk-1))'; m.data(nx048(nx)+H+33+40*(kk-1))';...
                                                         m.data(nx048(nx)+H+34+40*(kk-1))'; m.data(nx048(nx)+H+35+40*(kk-1))'],[],1),'double');
                    true_dop(idx,nx) = typecast(reshape([m.data(nx048(nx)+H+36+40*(kk-1))'; m.data(nx048(nx)+H+37+40*(kk-1))';...
                                                         m.data(nx048(nx)+H+38+40*(kk-1))'; m.data(nx048(nx)+H+39+40*(kk-1))';...
                                                         m.data(nx048(nx)+H+40+40*(kk-1))'; m.data(nx048(nx)+H+41+40*(kk-1))';...
                                                         m.data(nx048(nx)+H+42+40*(kk-1))'; m.data(nx048(nx)+H+43+40*(kk-1))'],[],1),'double');
                    app_dop(idx,nx)  = typecast(reshape([m.data(nx048(nx)+H+44+40*(kk-1))'; m.data(nx048(nx)+H+45+40*(kk-1))';...
                                                         m.data(nx048(nx)+H+46+40*(kk-1))'; m.data(nx048(nx)+H+47+40*(kk-1))';...
                                                         m.data(nx048(nx)+H+48+40*(kk-1))'; m.data(nx048(nx)+H+49+40*(kk-1))';...
                                                         m.data(nx048(nx)+H+50+40*(kk-1))'; m.data(nx048(nx)+H+51+40*(kk-1))'],[],1),'double');
            end
        end
    end
end
%%
% get TIME log index, ID: 101, or uint8([101 0])
nxrx = find(m.data(nxrec+4)==101 & m.data(nxrec+5)==0);
if ~isempty(nxrx)
    fprintf('TIME...')
    nx101 = nxrec(nxrx);
    msgln101 = typecast(reshape([m.data(nx101+8)'; m.data(nx101+9)'],[],1),'uint16');
    wk101 = typecast(reshape([m.data(nx101+14)'; m.data(nx101+15)'],[],1),'uint16');
    ms101 = typecast(reshape([m.data(nx101+16)'; m.data(nx101+17)';...
        m.data(nx101+18)'; m.data(nx101+19)'],[],1),'uint32');
    time_offset = typecast(reshape([m.data(nx101+H+4)'; m.data(nx101+H+5)';...
        m.data(nx101+H+6)'; m.data(nx101+H+7)'; m.data(nx101+H+8)'; m.data(nx101+H+9)';...
        m.data(nx101+H+10)'; m.data(nx101+H+11)'],[],1),'double');
    time_offset_std = typecast(reshape([m.data(nx101+H+12)'; m.data(nx101+H+13)';...
        m.data(nx101+H+14)'; m.data(nx101+H+15)'; m.data(nx101+H+16)'; m.data(nx101+H+17)';...
        m.data(nx101+H+18)'; m.data(nx101+H+19)'],[],1),'double');
end

%%
% get INSCONFIG log index, ID: 1945, or uint8([153 7])
nxrx = find(m.data(nxrec+4)==153 & m.data(nxrec+5)==7);
if ~isempty(nxrx) && parse_ins
    fprintf('INSCONFIG (first instance)...')
    nx1945 = nxrec(nxrx(1));
    msgln1945 = typecast(reshape([m.data(nx1945+8)'; m.data(nx1945+9)'],[],1),'uint16');
    wk1945 = typecast(reshape([m.data(nx1945+14)'; m.data(nx1945+15)'],[],1),'uint16');
    ms1945 = typecast(reshape([m.data(nx1945+16)'; m.data(nx1945+17)';...
        m.data(nx1945+18)'; m.data(nx1945+19)'],[],1),'uint32');
    sow1945 = double(ms1945)./1000;
    imutype = typecast(reshape([m.data(nx1945+H)'; m.data(nx1945+H+1)';...
        m.data(nx1945+H+2)'; m.data(nx1945+H+3)'],[],1),'uint32');
    initalignvel = typecast(reshape([m.data(nx1945+H+5)'],[],1),'uint8');
    imuprofile = typecast(reshape([m.data(nx1945+H+8)'; m.data(nx1945+H+9)';...
        m.data(nx1945+H+10)'; m.data(nx1945+H+11)'],[],1),'uint32');
    insupdates = typecast(reshape([m.data(nx1945+H+12)'; m.data(nx1945+H+13)';...
        m.data(nx1945+H+14)'; m.data(nx1945+H+15)'],[],1),'uint32');
    alignmentmode = typecast(reshape([m.data(nx1945+H+16)'; m.data(nx1945+H+17)';...
        m.data(nx1945+H+18)'; m.data(nx1945+H+19)'],[],1),'uint32');
    ntrans = typecast(reshape([m.data(nx1945+H+60)'; m.data(nx1945+H+61)';...
        m.data(nx1945+H+62)'; m.data(nx1945+H+63)'],[],1),'uint32');
    transvecs = zeros(ntrans,9);
    for k = 1:ntrans
        transvecs(k,1) = typecast(reshape([m.data(nx1945+H+(k-1)*36+64)'; m.data(nx1945+H+(k-1)*36+65)';...
            m.data(nx1945+H+(k-1)*36+66)'; m.data(nx1945+H+(k-1)*36+67)'],[],1),'uint32');
        transvecs(k,2) = typecast(reshape([m.data(nx1945+H+(k-1)*36+68)'; m.data(nx1945+H+(k-1)*36+69)';...
            m.data(nx1945+H+(k-1)*36+70)'; m.data(nx1945+H+(k-1)*36+71)'],[],1),'uint32');
        transvecs(k,3) = typecast(reshape([m.data(nx1945+H+(k-1)*36+72)'; m.data(nx1945+H+(k-1)*36+73)';...
            m.data(nx1945+H+(k-1)*36+74)'; m.data(nx1945+H+(k-1)*36+75)'],[],1),'single');
        transvecs(k,4) = typecast(reshape([m.data(nx1945+H+(k-1)*36+76)'; m.data(nx1945+H+(k-1)*36+77)';...
            m.data(nx1945+H+(k-1)*36+78)'; m.data(nx1945+H+(k-1)*36+79)'],[],1),'single');
        transvecs(k,5) = typecast(reshape([m.data(nx1945+H+(k-1)*36+80)'; m.data(nx1945+H+(k-1)*36+81)';...
            m.data(nx1945+H+(k-1)*36+82)'; m.data(nx1945+H+(k-1)*36+83)'],[],1),'single');
        transvecs(k,6) = typecast(reshape([m.data(nx1945+H+(k-1)*36+84)'; m.data(nx1945+H+(k-1)*36+85)';...
            m.data(nx1945+H+(k-1)*36+86)'; m.data(nx1945+H+(k-1)*36+87)'],[],1),'single');
        transvecs(k,7) = typecast(reshape([m.data(nx1945+H+(k-1)*36+88)'; m.data(nx1945+H+(k-1)*36+89)';...
            m.data(nx1945+H+(k-1)*36+90)'; m.data(nx1945+H+(k-1)*36+91)'],[],1),'single');
        transvecs(k,8) = typecast(reshape([m.data(nx1945+H+(k-1)*36+92)'; m.data(nx1945+H+(k-1)*36+93)';...
            m.data(nx1945+H+(k-1)*36+94)'; m.data(nx1945+H+(k-1)*36+95)'],[],1),'single');
        transvecs(k,9) = typecast(reshape([m.data(nx1945+H+(k-1)*36+96)'; m.data(nx1945+H+(k-1)*36+97)';...
            m.data(nx1945+H+(k-1)*36+98)'; m.data(nx1945+H+(k-1)*36+99)'],[],1),'uint32');
    end
    nrots = typecast(reshape([m.data(nx1945+H+ntrans*36+64)'; m.data(nx1945+H+ntrans*36+65)';...
        m.data(nx1945+H+ntrans*36+66)'; m.data(nx1945+H+ntrans*36+67)'],[],1),'uint32');
    rotvecs = zeros(nrots,9);
    for k = 1:nrots
        rotvecs(k,1) = typecast(reshape([m.data(nx1945+H+(ntrans+k-1)*36+68)'; m.data(nx1945+H+(ntrans+k-1)*36+69)';...
            m.data(nx1945+H+(ntrans+k-1)*36+70)'; m.data(nx1945+H+(ntrans+k-1)*36+71)'],[],1),'uint32');
        rotvecs(k,2) = typecast(reshape([m.data(nx1945+H+(ntrans+k-1)*36+72)'; m.data(nx1945+H+(ntrans+k-1)*36+73)';...
            m.data(nx1945+H+(ntrans+k-1)*36+74)'; m.data(nx1945+H+(ntrans+k-1)*36+75)'],[],1),'single');
        rotvecs(k,3) = typecast(reshape([m.data(nx1945+H+(ntrans+k-1)*36+76)'; m.data(nx1945+H+(ntrans+k-1)*36+77)';...
            m.data(nx1945+H+(ntrans+k-1)*36+78)'; m.data(nx1945+H+(ntrans+k-1)*36+79)'],[],1),'single');
        rotvecs(k,4) = typecast(reshape([m.data(nx1945+H+(ntrans+k-1)*36+80)'; m.data(nx1945+H+(ntrans+k-1)*36+81)';...
            m.data(nx1945+H+(ntrans+k-1)*36+82)'; m.data(nx1945+H+(ntrans+k-1)*36+83)'],[],1),'single');
        rotvecs(k,5) = typecast(reshape([m.data(nx1945+H+(ntrans+k-1)*36+84)'; m.data(nx1945+H+(ntrans+k-1)*36+85)';...
            m.data(nx1945+H+(ntrans+k-1)*36+86)'; m.data(nx1945+H+(ntrans+k-1)*36+87)'],[],1),'single');
        rotvecs(k,6) = typecast(reshape([m.data(nx1945+H+(ntrans+k-1)*36+88)'; m.data(nx1945+H+(ntrans+k-1)*36+89)';...
            m.data(nx1945+H+(ntrans+k-1)*36+90)'; m.data(nx1945+H+(ntrans+k-1)*36+91)'],[],1),'single');
        rotvecs(k,7) = typecast(reshape([m.data(nx1945+H+(ntrans+k-1)*36+92)'; m.data(nx1945+H+(ntrans+k-1)*36+93)';...
            m.data(nx1945+H+(ntrans+k-1)*36+94)'; m.data(nx1945+H+(ntrans+k-1)*36+95)'],[],1),'single');
        rotvecs(k,8) = typecast(reshape([m.data(nx1945+H+(ntrans+k-1)*36+96)'; m.data(nx1945+H+(ntrans+k-1)*36+97)';...
            m.data(nx1945+H+(ntrans+k-1)*36+98)'; m.data(nx1945+H+(ntrans+k-1)*36+99)'],[],1),'single');
        rotvecs(k,9) = typecast(reshape([m.data(nx1945+H+(ntrans+k-1)*36+100)'; m.data(nx1945+H+(ntrans+k-1)*36+101)';...
            m.data(nx1945+H+(ntrans+k-1)*36+102)'; m.data(nx1945+H+(ntrans+k-1)*36+103)'],[],1),'uint32');
    end
end

%%
% get INSPVAX log index, ID: 1465, or uint8([185 5])
nxrx = find(m.data(nxrec+4)==185 & m.data(nxrec+5)==5);
if ~isempty(nxrx) && parse_ins
    fprintf('INSPVAX...')
    nx1465 = nxrec(nxrx);
    msgln1465 = typecast(reshape([m.data(nx1465+8)'; m.data(nx1465+9)'],[],1),'uint16');
    wk1465 = typecast(reshape([m.data(nx1465+14)'; m.data(nx1465+15)'],[],1),'uint16');
    ms1465 = typecast(reshape([m.data(nx1465+16)'; m.data(nx1465+17)';...
        m.data(nx1465+18)'; m.data(nx1465+19)'],[],1),'uint32');
    sow1465 = double(ms1465)./1000;
    insstat = typecast(reshape([m.data(nx1465+H)'; m.data(nx1465+H+1)';...
        m.data(nx1465+H+2)'; m.data(nx1465+H+3)'],[],1),'uint32');
    inslat = typecast(reshape([m.data(nx1465+H+8)'; m.data(nx1465+H+9)';...
        m.data(nx1465+H+10)'; m.data(nx1465+H+11)'; m.data(nx1465+H+12)'; m.data(nx1465+H+13)';...
        m.data(nx1465+H+14)'; m.data(nx1465+H+15)'],[],1),'double');
    inslon = typecast(reshape([m.data(nx1465+H+16)'; m.data(nx1465+H+17)';...
        m.data(nx1465+H+18)'; m.data(nx1465+H+19)'; m.data(nx1465+H+20)'; m.data(nx1465+H+21)';...
        m.data(nx1465+H+22)'; m.data(nx1465+H+23)'],[],1),'double');
    inshgt = typecast(reshape([m.data(nx1465+H+24)'; m.data(nx1465+H+25)';...
        m.data(nx1465+H+26)'; m.data(nx1465+H+27)'; m.data(nx1465+H+28)'; m.data(nx1465+H+29)';...
        m.data(nx1465+H+30)'; m.data(nx1465+H+31)'],[],1),'double');
    insght = typecast(reshape([m.data(nx1465+H+32)'; m.data(nx1465+H+33)';...
        m.data(nx1465+H+34)'; m.data(nx1465+H+35)'],[],1),'single');
    insnorthvel = typecast(reshape([m.data(nx1465+H+36)'; m.data(nx1465+H+37)';...
        m.data(nx1465+H+38)'; m.data(nx1465+H+39)'; m.data(nx1465+H+40)'; m.data(nx1465+H+41)';...
        m.data(nx1465+H+42)'; m.data(nx1465+H+43)'],[],1),'double');
    inseastvel = typecast(reshape([m.data(nx1465+H+44)'; m.data(nx1465+H+45)';...
        m.data(nx1465+H+46)'; m.data(nx1465+H+47)'; m.data(nx1465+H+48)'; m.data(nx1465+H+49)';...
        m.data(nx1465+H+50)'; m.data(nx1465+H+51)'],[],1),'double');
    insupvel = typecast(reshape([m.data(nx1465+H+52)'; m.data(nx1465+H+53)';...
        m.data(nx1465+H+54)'; m.data(nx1465+H+55)'; m.data(nx1465+H+56)'; m.data(nx1465+H+57)';...
        m.data(nx1465+H+58)'; m.data(nx1465+H+59)'],[],1),'double');
    insroll = typecast(reshape([m.data(nx1465+H+60)'; m.data(nx1465+H+61)';...
        m.data(nx1465+H+62)'; m.data(nx1465+H+63)'; m.data(nx1465+H+64)'; m.data(nx1465+H+65)';...
        m.data(nx1465+H+66)'; m.data(nx1465+H+67)'],[],1),'double');
    inspitch = typecast(reshape([m.data(nx1465+H+68)'; m.data(nx1465+H+69)';...
        m.data(nx1465+H+70)'; m.data(nx1465+H+71)'; m.data(nx1465+H+72)'; m.data(nx1465+H+73)';...
        m.data(nx1465+H+74)'; m.data(nx1465+H+75)'],[],1),'double');
    insazim = typecast(reshape([m.data(nx1465+H+76)'; m.data(nx1465+H+77)';...
        m.data(nx1465+H+78)'; m.data(nx1465+H+79)'; m.data(nx1465+H+80)'; m.data(nx1465+H+81)';...
        m.data(nx1465+H+82)'; m.data(nx1465+H+83)'],[],1),'double');
    insrollstd = typecast(reshape([m.data(nx1465+H+108)'; m.data(nx1465+H+109)';...
        m.data(nx1465+H+110)'; m.data(nx1465+H+111)'],[],1),'double');
    inspitchstd = typecast(reshape([m.data(nx1465+H+112)'; m.data(nx1465+H+113)';...
        m.data(nx1465+H+114)'; m.data(nx1465+H+115)'],[],1),'double');
    insazimstd = typecast(reshape([m.data(nx1465+H+116)'; m.data(nx1465+H+117)';...
        m.data(nx1465+H+118)'; m.data(nx1465+H+119)'],[],1),'double');
end

%%
% get INSUPDATESTATUS log index, ID: 1825, or uint8([33 7])
nxrx = find(m.data(nxrec+4)==33 & m.data(nxrec+5)==7);
% gotinsupdate = 0;
if ~isempty(nxrx) && parse_ins
    fprintf('INSUPDATESTATUS...')
    nx1825 = nxrec(nxrx);
    msgln1825 = typecast(reshape([m.data(nx1825+8)'; m.data(nx1825+9)'],[],1),'uint16');
    wk1825 = typecast(reshape([m.data(nx1825+14)'; m.data(nx1825+15)'],[],1),'uint16');
    ms1825 = typecast(reshape([m.data(nx1825+16)'; m.data(nx1825+17)';...
        m.data(nx1825+18)'; m.data(nx1825+19)'],[],1),'uint32');
    sow1825 = double(ms1825)./1000;
    
    inspostype = typecast(reshape([m.data(nx1825+H)'; m.data(nx1825+H+1)';...
        m.data(nx1825+H+2)'; m.data(nx1825+H+3)'],[],1),'uint32');
    insnumpsr = typecast(reshape([m.data(nx1825+H+4)'; m.data(nx1825+H+5)';...
        m.data(nx1825+H+6)'; m.data(nx1825+H+7)'],[],1),'uint32');
    insnumadr = typecast(reshape([m.data(nx1825+H+8)'; m.data(nx1825+H+9)';...
        m.data(nx1825+H+10)'; m.data(nx1825+H+11)'],[],1),'uint32');
    insnumdop = typecast(reshape([m.data(nx1825+H+12)'; m.data(nx1825+H+13)';...
        m.data(nx1825+H+14)'; m.data(nx1825+H+15)'],[],1),'uint32');
    dmiupdate = typecast(reshape([m.data(nx1825+H+16)'; m.data(nx1825+H+17)';...
        m.data(nx1825+H+18)'; m.data(nx1825+H+19)'],[],1),'uint32');
    headingupdate = typecast(reshape([m.data(nx1825+H+20)'; m.data(nx1825+H+21)';...
        m.data(nx1825+H+22)'; m.data(nx1825+H+23)'],[],1),'uint32');
    extsolstat = typecast(reshape([m.data(nx1825+H+24)'; m.data(nx1825+H+25)';...
        m.data(nx1825+H+26)'; m.data(nx1825+H+27)'],[],1),'uint32');
    insenabledupdt = typecast(reshape([m.data(nx1825+H+28)'; m.data(nx1825+H+29)';...
        m.data(nx1825+H+30)'; m.data(nx1825+H+31)'],[],1),'uint32');
end
% gotinsupdate=1;

%%
% get RAWIMUSX log index, ID: 1462, or uint8([182 5])
nxsrx = find(m.data(nxsrec+4)==182 & m.data(nxsrec+5)==5);
if ~isempty(nxsrx) && parse_ins
    fprintf('RAWIMUSX...')
    nx1462 = nxsrec(nxsrx);
    msgln1462 = typecast(reshape([m.data(nx1462+3)'],[],1),'uint8');
    wk1462 = typecast(reshape([m.data(nx1462+6)'; m.data(nx1462+7)'],[],1),'uint16');
    ms1462 = typecast(reshape([m.data(nx1462+8)'; m.data(nx1462+9)';...
        m.data(nx1462+10)'; m.data(nx1462+11)'],[],1),'uint32');
    sow1462 = double(ms1462)./1000;
    rawimuinfo = typecast(reshape(m.data(nx1462+Hs)',[],1),'uint8');
    rawimutype = typecast(reshape(m.data(nx1462+Hs+1)',[],1),'uint8');
    rawimugnsswk = typecast(reshape([m.data(nx1462+Hs+2)'; m.data(nx1462+Hs+3)'],[],1),'uint16');
    rawimugnsssow = typecast(reshape([m.data(nx1462+Hs+4)'; m.data(nx1462+Hs+5)';...
        m.data(nx1462+Hs+6)'; m.data(nx1462+Hs+7)'; m.data(nx1462+Hs+8)';...
        m.data(nx1462+Hs+9)'; m.data(nx1462+Hs+10)'; m.data(nx1462+Hs+11)'],[],1),'double');
    rawimustatus = typecast(reshape([m.data(nx1462+Hs+12)'; m.data(nx1462+Hs+13)';...
        m.data(nx1462+Hs+14)'; m.data(nx1462+Hs+15)'],[],1),'uint32');
    rawimuzaccel = typecast(reshape([m.data(nx1462+Hs+16)'; m.data(nx1462+Hs+17)';...
        m.data(nx1462+Hs+18)'; m.data(nx1462+Hs+19)'],[],1),'uint32');
    rawimuyaccel = typecast(reshape([m.data(nx1462+Hs+20)'; m.data(nx1462+Hs+21)';...
        m.data(nx1462+Hs+22)'; m.data(nx1462+Hs+23)'],[],1),'uint32');
    rawimuxaccel = typecast(reshape([m.data(nx1462+Hs+24)'; m.data(nx1462+Hs+25)';...
        m.data(nx1462+Hs+26)'; m.data(nx1462+H+27)'],[],1),'uint32');
    rawimuzgyro = typecast(reshape([m.data(nx1462+Hs+28)'; m.data(nx1462+Hs+29)';...
        m.data(nx1462+Hs+30)'; m.data(nx1462+Hs+31)'],[],1),'uint32');
    rawimuygyro = typecast(reshape([m.data(nx1462+Hs+32)'; m.data(nx1462+Hs+33)';...
        m.data(nx1462+Hs+34)'; m.data(nx1462+Hs+35)'],[],1),'uint32');
    rawimuxgyro = typecast(reshape([m.data(nx1462+Hs+36)'; m.data(nx1462+Hs+37)';...
        m.data(nx1462+Hs+38)'; m.data(nx1462+Hs+39)'],[],1),'uint32');
end

%%
% get HEADING2 log index, ID: 1335, or uint8([55 5])
nxrx = find(m.data(nxrec+4)==55 & m.data(nxrec+5)==5);
if ~isempty(nxrx) && parse_bestpos
    fprintf('HEADING2...')
    nx1335 = nxrec(nxrx);
    msgln1335 = typecast(reshape([m.data(nx1335+8)'; m.data(nx1335+9)'],[],1),'uint16');
    wk1335 = typecast(reshape([m.data(nx1335+14)'; m.data(nx1335+15)'],[],1),'uint16');
    ms1335 = typecast(reshape([m.data(nx1335+16)'; m.data(nx1335+17)';...
        m.data(nx1335+18)'; m.data(nx1335+19)'],[],1),'uint32');
    sow1335 = double(ms1335)./1000;   
    hdgsolstat = typecast(reshape([m.data(nx1335+H)'; m.data(nx1335+H+1)';...
        m.data(nx1335+H+2)'; m.data(nx1335+H+3)'],[],1),'uint32');
    hdgpostype = typecast(reshape([m.data(nx1335+H+4)'; m.data(nx1335+H+5)';...
        m.data(nx1335+H+6)'; m.data(nx1335+H+7)'],[],1),'uint32');  
    hdglength = typecast(reshape([m.data(nx1335+H+8)'; m.data(nx1335+H+9)';...
        m.data(nx1335+H+10)'; m.data(nx1335+H+11)'],[],1),'single');
    heading = typecast(reshape([m.data(nx1335+H+12)'; m.data(nx1335+H+13)';...
        m.data(nx1335+H+14)'; m.data(nx1335+H+15)'],[],1),'single');
    pitch = typecast(reshape([m.data(nx1335+H+16)'; m.data(nx1335+H+17)';...
        m.data(nx1335+H+18)'; m.data(nx1335+H+19)'],[],1),'single');
    headingstd = typecast(reshape([m.data(nx1335+H+24)'; m.data(nx1335+H+25)';...
        m.data(nx1335+H+26)'; m.data(nx1335+H+27)'],[],1),'single');
    pitchstd = typecast(reshape([m.data(nx1335+H+28)'; m.data(nx1335+H+29)';...
        m.data(nx1335+H+30)'; m.data(nx1335+H+31)'],[],1),'single');
    roverid = typecast(reshape([m.data(nx1335+H+32)'; m.data(nx1335+H+33)';...
        m.data(nx1335+H+34)'; m.data(nx1335+H+35)'],[],1),'uint32');
    masterid = typecast(reshape([m.data(nx1335+H+36)'; m.data(nx1335+H+37)';...
        m.data(nx1335+H+38)'; m.data(nx1335+H+39)'],[],1),'uint32');
    hdgsats = typecast(reshape(m.data(nx1335+H+40)',[],1),'uint8');
    hdgsolnSV = typecast(reshape(m.data(nx1335+H+41)',[],1),'uint8');
    hdgobs = typecast(reshape(m.data(nx1335+H+42)',[],1),'uint8');
    hdgmulti = typecast(reshape(m.data(nx1335+H+43)',[],1),'uint8');
    hdgsolsrc = typecast(reshape(m.data(nx1335+H+44)',[],1),'uint8');
    hdgextsolstat = typecast(reshape(m.data(nx1335+H+45)',[],1),'uint8');
    hdggalbdmask = typecast(reshape(m.data(nx1335+H+46)',[],1),'uint8');
    hdggpsglomask = typecast(reshape(m.data(nx1335+H+47)',[],1),'uint8');

end
%%
% get BESTPOS log index, ID: 42, or uint8([42 0])
nxrx = find(m.data(nxrec+4)==42 & m.data(nxrec+5)==0);
if ~isempty(nxrx) && parse_bestpos
    fprintf('BESTPOS...')
    nx42 = nxrec(nxrx);
    msgln42 = typecast(reshape([m.data(nx42+8)'; m.data(nx42+9)'],[],1),'uint16');
    wk42 = typecast(reshape([m.data(nx42+14)'; m.data(nx42+15)'],[],1),'uint16');
    ms42 = typecast(reshape([m.data(nx42+16)'; m.data(nx42+17)';...
        m.data(nx42+18)'; m.data(nx42+19)'],[],1),'uint32');
    sow42 = double(ms42)./1000;
    bpsolstat = typecast(reshape([m.data(nx42+H)'; m.data(nx42+H+1)';...
        m.data(nx42+H+2)'; m.data(nx42+H+3)'],[],1),'uint32');
    bppostype = typecast(reshape([m.data(nx42+H+4)'; m.data(nx42+H+5)';...
        m.data(nx42+H+6)'; m.data(nx42+H+7)'],[],1),'uint32');
    bplat = typecast(reshape([m.data(nx42+H+8)'; m.data(nx42+H+9)';...
        m.data(nx42+H+10)'; m.data(nx42+H+11)'; m.data(nx42+H+12)'; m.data(nx42+H+13)';...
        m.data(nx42+H+14)'; m.data(nx42+H+15)'],[],1),'double');
    bplon = typecast(reshape([m.data(nx42+H+16)'; m.data(nx42+H+17)';...
        m.data(nx42+H+18)'; m.data(nx42+H+19)'; m.data(nx42+H+20)'; m.data(nx42+H+21)';...
        m.data(nx42+H+22)'; m.data(nx42+H+23)'],[],1),'double');
    bphgt = typecast(reshape([m.data(nx42+H+24)'; m.data(nx42+H+25)';...
        m.data(nx42+H+26)'; m.data(nx42+H+27)'; m.data(nx42+H+28)'; m.data(nx42+H+29)';...
        m.data(nx42+H+30)'; m.data(nx42+H+31)'],[],1),'double');
    bpght = typecast(reshape([m.data(nx42+H+32)'; m.data(nx42+H+33)';...
        m.data(nx42+H+34)'; m.data(nx42+H+35)'],[],1),'single');
    bpdatum_id = typecast(reshape([m.data(nx42(end)+H+36)'; m.data(nx42(end)+H+37)';...
        m.data(nx42(end)+H+38)'; m.data(nx42(end)+H+39)'],[],1),'uint32');
    bplatstd = typecast(reshape([m.data(nx42+H+40)'; m.data(nx42+H+41)';...
        m.data(nx42+H+42)'; m.data(nx42+H+43)'],[],1),'single');
    bplonstd = typecast(reshape([m.data(nx42+H+44)'; m.data(nx42+H+45)';...
        m.data(nx42+H+46)'; m.data(nx42+H+47)'],[],1),'single');
    bphgtstd = typecast(reshape([m.data(nx42+H+48)'; m.data(nx42+H+49)';...
        m.data(nx42+H+50)'; m.data(nx42+H+51)'],[],1),'single');
end


%%
% get PPPPOS log index, ID: 1538, or uint8([2 6])
nxrx = find(m.data(nxrec+4)==2 & m.data(nxrec+5)==6);
if ~isempty(nxrx) & parse_ppp
    fprintf('PPPPOS...')
    nx1538 = nxrec(nxrx);
    msgln1538 = typecast(reshape([m.data(nx1538+8)'; m.data(nx1538+9)'],[],1),'uint16');
    wk1538 = typecast(reshape([m.data(nx1538+14)'; m.data(nx1538+15)'],[],1),'uint16');
    ms1538 = typecast(reshape([m.data(nx1538+16)'; m.data(nx1538+17)';...
        m.data(nx1538+18)'; m.data(nx1538+19)'],[],1),'uint32');
    sow1538 = double(ms1538)./1000;
    pppstatus = typecast(reshape([m.data(nx1538+H)'; m.data(nx1538+H+1)';...
        m.data(nx1538+H+2)'; m.data(nx1538+H+3)'],[],1),'uint32');
    ppppostype = typecast(reshape([m.data(nx1538+H+4)'; m.data(nx1538+H+5)';...
        m.data(nx1538+H+6)'; m.data(nx1538+H+7)'],[],1),'uint32');
    ppplat = typecast(reshape([m.data(nx1538+H+8)'; m.data(nx1538+H+9)';...
        m.data(nx1538+H+10)'; m.data(nx1538+H+11)'; m.data(nx1538+H+12)'; m.data(nx1538+H+13)';...
        m.data(nx1538+H+14)'; m.data(nx1538+H+15)'],[],1),'double');
    ppplon = typecast(reshape([m.data(nx1538+H+16)'; m.data(nx1538+H+17)';...
        m.data(nx1538+H+18)'; m.data(nx1538+H+19)'; m.data(nx1538+H+20)'; m.data(nx1538+H+21)';...
        m.data(nx1538+H+22)'; m.data(nx1538+H+23)'],[],1),'double');
    ppphgt = typecast(reshape([m.data(nx1538+H+24)'; m.data(nx1538+H+25)';...
        m.data(nx1538+H+26)'; m.data(nx1538+H+27)'; m.data(nx1538+H+28)'; m.data(nx1538+H+29)';...
        m.data(nx1538+H+30)'; m.data(nx1538+H+31)'],[],1),'double');
    pppundulation = typecast(reshape([m.data(nx1538+H+32)'; m.data(nx1538+H+33)';...
        m.data(nx1538+H+34)'; m.data(nx1538+H+35)'],[],1),'single');
    pppdatum_id = typecast(reshape([m.data(nx1538+H+36)'; m.data(nx1538+H+37)';...
        m.data(nx1538+H+38)'; m.data(nx1538+H+39)'],[],1),'uint32');  
    ppplatstd = typecast(reshape([m.data(nx1538+H+40)'; m.data(nx1538+H+41)';...
        m.data(nx1538+H+42)'; m.data(nx1538+H+43)'],[],1),'single');
    ppplonstd = typecast(reshape([m.data(nx1538+H+44)'; m.data(nx1538+H+45)';...
        m.data(nx1538+H+46)'; m.data(nx1538+H+47)'],[],1),'single');
    ppphgtstd = typecast(reshape([m.data(nx1538+H+48)'; m.data(nx1538+H+49)';...
        m.data(nx1538+H+50)'; m.data(nx1538+H+51)'],[],1),'single');    
end


%%
% get PSRPOS log index, ID: 47, or uint8([47 0])
nxrx = find(m.data(nxrec+4)==47 & m.data(nxrec+5)==0);
if ~isempty(nxrx) && parse_psrpos
    fprintf('PSRPOS...')
    nx47 = nxrec(nxrx);
    msgln47 = typecast(reshape([m.data(nx47+8)'; m.data(nx47+9)'],[],1),'uint16');
    wk47 = typecast(reshape([m.data(nx47+14)'; m.data(nx47+15)'],[],1),'uint16');
    ms47 = typecast(reshape([m.data(nx47+16)'; m.data(nx47+17)';...
        m.data(nx47+18)'; m.data(nx47+19)'],[],1),'uint32');
    sow47 = double(ms47)./1000;
    psrsolstat = typecast(reshape([m.data(nx47+H)'; m.data(nx47+H+1)';...
        m.data(nx47+H+2)'; m.data(nx47+H+3)'],[],1),'uint32');
    psrpostype = typecast(reshape([m.data(nx47+H+4)'; m.data(nx47+H+5)';...
        m.data(nx47+H+6)'; m.data(nx47+H+7)'],[],1),'uint32');
    psrlat = typecast(reshape([m.data(nx47+H+8)'; m.data(nx47+H+9)';...
        m.data(nx47+H+10)'; m.data(nx47+H+11)'; m.data(nx47+H+12)'; m.data(nx47+H+13)';...
        m.data(nx47+H+14)'; m.data(nx47+H+15)'],[],1),'double');
    psrlon = typecast(reshape([m.data(nx47+H+16)'; m.data(nx47+H+17)';...
        m.data(nx47+H+18)'; m.data(nx47+H+19)'; m.data(nx47+H+20)'; m.data(nx47+H+21)';...
        m.data(nx47+H+22)'; m.data(nx47+H+23)'],[],1),'double');
    psrhgt = typecast(reshape([m.data(nx47+H+24)'; m.data(nx47+H+25)';...
        m.data(nx47+H+26)'; m.data(nx47+H+27)'; m.data(nx47+H+28)'; m.data(nx47+H+29)';...
        m.data(nx47+H+30)'; m.data(nx47+H+31)'],[],1),'double');
    psrundulation = typecast(reshape([m.data(nx47+H+32)'; m.data(nx47+H+33)';...
        m.data(nx47+H+34)'; m.data(nx47+H+35)'],[],1),'single');
    psrdatum_id = typecast(reshape([m.data(nx47(end)+H+36)'; m.data(nx47(end)+H+37)';...
        m.data(nx47(end)+H+38)'; m.data(nx47(end)+H+39)'],[],1),'uint32');
    psrlatstd = typecast(reshape([m.data(nx47+H+40)'; m.data(nx47+H+41)';...
        m.data(nx47+H+42)'; m.data(nx47+H+43)'],[],1),'single');
    psrlonstd = typecast(reshape([m.data(nx47+H+44)'; m.data(nx47+H+45)';...
        m.data(nx47+H+46)'; m.data(nx47+H+47)'],[],1),'single');
    psrhgtstd = typecast(reshape([m.data(nx47+H+48)'; m.data(nx47+H+49)';...
        m.data(nx47+H+50)'; m.data(nx47+H+51)'],[],1),'single');
end


%%
% get BESTGNSSVEL log index, ID: 1430, or uint8([150 5])
nxrx = find(m.data(nxrec+4)==150 & m.data(nxrec+5)==5);
if ~isempty(nxrx) && parse_vel
    fprintf('BESTGNSSVEL...')
    nx1430 = nxrec(nxrx);
    msgln1430 = typecast(reshape([m.data(nx1430+8)'; m.data(nx1430+9)'],[],1),'uint16');
    wk1430 = typecast(reshape([m.data(nx1430+14)'; m.data(nx1430+15)'],[],1),'uint16');
    ms1430 = typecast(reshape([m.data(nx1430+16)'; m.data(nx1430+17)';...
        m.data(nx1430+18)'; m.data(nx1430+19)'],[],1),'uint32');
    sow1430 = double(ms1430)./1000;
    gnssvelsolstat = typecast(reshape([m.data(nx1430+H)'; m.data(nx1430+H+1)';...
        m.data(nx1430+H+2)'; m.data(nx1430+H+3)'],[],1),'uint32');
    gnssveltype = typecast(reshape([m.data(nx1430+H+4)'; m.data(nx1430+H+5)';...
        m.data(nx1430+H+6)'; m.data(nx1430+H+7)'],[],1),'uint32');
    gnssvellatency = typecast(reshape([m.data(nx1430+H+8)'; m.data(nx1430+H+9)';...
        m.data(nx1430+H+10)'; m.data(nx1430+H+11)'],[],1),'single');
    gnssvelage = typecast(reshape([m.data(nx1430+H+12)'; m.data(nx1430+H+13)';...
        m.data(nx1430+H+14)'; m.data(nx1430+H+15)'],[],1),'single');    
    gnssvelhorspd = typecast(reshape([m.data(nx1430+H+16)'; m.data(nx1430+H+17)';...
        m.data(nx1430+H+18)'; m.data(nx1430+H+19)'; m.data(nx1430+H+20)';...
        m.data(nx1430+H+21)'; m.data(nx1430+H+22)'; m.data(nx1430+H+23)'],[],1),'double');
    gnssveltrkgnd = typecast(reshape([m.data(nx1430+H+24)'; m.data(nx1430+H+25)';...
        m.data(nx1430+H+26)'; m.data(nx1430+H+27)'; m.data(nx1430+H+28)';...
        m.data(nx1430+H+29)'; m.data(nx1430+H+30)'; m.data(nx1430+H+31)'],[],1),'double');
    gnssvelvertspd = typecast(reshape([m.data(nx1430+H+32)'; m.data(nx1430+H+33)';...
        m.data(nx1430+H+34)'; m.data(nx1430+H+35)'; m.data(nx1430+H+36)';...
        m.data(nx1430+H+37)'; m.data(nx1430+H+38)'; m.data(nx1430+H+39)'],[],1),'double');
end

%%
% get BESTVEL log index, ID: 99, or uint8([99 0])
nxrx = find(m.data(nxrec+4)==99 & m.data(nxrec+5)==0);
if ~isempty(nxrx) && parse_vel
    fprintf('BESTVEL...')
    nx99 = nxrec(nxrx);
    msgln99 = typecast(reshape([m.data(nx99+8)'; m.data(nx99+9)'],[],1),'uint16');
    wk99 = typecast(reshape([m.data(nx99+14)'; m.data(nx99+15)'],[],1),'uint16');
    ms99 = typecast(reshape([m.data(nx99+16)'; m.data(nx99+17)';...
        m.data(nx99+18)'; m.data(nx99+19)'],[],1),'uint32');
    sow99 = double(ms99)./1000;
    bestvelsolstat = typecast(reshape([m.data(nx99+H)'; m.data(nx99+H+1)';...
        m.data(nx99+H+2)'; m.data(nx99+H+3)'],[],1),'uint32');
    bestveltype = typecast(reshape([m.data(nx99+H+4)'; m.data(nx99+H+5)';...
        m.data(nx99+H+6)'; m.data(nx99+H+7)'],[],1),'uint32');
    bestvellatency = typecast(reshape([m.data(nx99+H+8)'; m.data(nx99+H+9)';...
        m.data(nx99+H+10)'; m.data(nx99+H+11)'],[],1),'single');
    bestvelage = typecast(reshape([m.data(nx99+H+12)'; m.data(nx99+H+13)';...
        m.data(nx99+H+14)'; m.data(nx99+H+15)'],[],1),'single');    
    bestvelhorspd = typecast(reshape([m.data(nx99+H+16)'; m.data(nx99+H+17)';...
        m.data(nx99+H+18)'; m.data(nx99+H+19)'; m.data(nx99+H+20)';...
        m.data(nx99+H+21)'; m.data(nx99+H+22)'; m.data(nx99+H+23)'],[],1),'double');
    bestveltrkgnd = typecast(reshape([m.data(nx99+H+24)'; m.data(nx99+H+25)';...
        m.data(nx99+H+26)'; m.data(nx99+H+27)'; m.data(nx99+H+28)';...
        m.data(nx99+H+29)'; m.data(nx99+H+30)'; m.data(nx99+H+31)'],[],1),'double');
    bestvelvertspd = typecast(reshape([m.data(nx99+H+32)'; m.data(nx99+H+33)';...
        m.data(nx99+H+34)'; m.data(nx99+H+35)'; m.data(nx99+H+36)';...
        m.data(nx99+H+37)'; m.data(nx99+H+38)'; m.data(nx99+H+39)'],[],1),'double');
end

fprintf(' done.\n');

%%
if C1L1P2L2_like_readobs
    fprintf('Reformatting C1, L1, P2, L2 to be like readobs at %d-sec interval.\n',decimation_interval);
    disp('(Original matrices are in C1o, L1o, P2o and L2o)');
    decnx = find(mod(round(sow_rge),decimation_interval)==0);
    PRN_rge1d = PRN_rge1(:,decnx);
    PRN_rge2d = PRN_rge2(:,decnx);
    osat = ((PRN_rge1d(1:size(PRN_rge2d,1),:)+PRN_rge2d)./2)';
    owk = double(wk043(decnx,1)); owk(owk<1024)=owk(owk<1024)+1024;
    osow = sow_rge(decnx,1);
    [oyr, omo, ody, ohr, omin, osec] = datevec(7*owk+osow./86400+datenum('01/06/1980'));
    % put original observations into matrices with "o" suffix:
    C1o = C1; L1o = L1; CNo1o = CNo1; P2o = P2; L2o = L2; CNo2o = CNo2;
    locktime1o = locktime1; locktime2o = locktime2;
    % make decimated versions:
    C1d = C1(:,decnx); L1d = L1(:,decnx); P2d = P2(:,decnx); L2d = L2(:,decnx);
    CNo1d = CNo1(:,decnx); CNo2d = CNo2(:,decnx);
    locktime1d = locktime1(:,decnx); locktime2d = locktime2(:,decnx);
    maxprn = max(PRN_rge2d(~isnan(PRN_rge2d)));
    if maxprn > 32
        maxprn = 32;
    end
    C1 = NaN.*zeros(length(decnx),maxprn);
    L1 = NaN.*zeros(length(decnx),maxprn);
    CNo1 = NaN.*zeros(length(decnx),maxprn);
    locktime1 = NaN.*zeros(length(decnx),maxprn);
    P2 = NaN.*zeros(length(decnx),maxprn);
    L2 = NaN.*zeros(length(decnx),maxprn);
    CNo2 = NaN.*zeros(length(decnx),maxprn);
    locktime2 = NaN.*zeros(length(decnx),maxprn);
    for k = 1:maxprn
        nx = find(PRN_rge1d == k);
        [~,nxc] = find(PRN_rge1d == k);
        C1(nxc,k) = C1d(nx);
        L1(nxc,k) = -L1d(nx);
        CNo1(nxc,k) = CNo1d(nx);
        locktime1(nxc,k) = locktime1d(nx);
        nx = find(PRN_rge2d == k);
        [nxr,nxc] = find(PRN_rge2d == k);
        P2(nxc,k) = P2d(nx);
        L2(nxc,k) = -L2d(nx);
        CNo2(nxc,k) = CNo2d(nx);
        locktime2(nxc,k) = locktime2d(nx);
    end
end


