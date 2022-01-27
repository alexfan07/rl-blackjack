% 2021/12/07 written by Fan Yifeng
% Homework 2: Blackjack

clear;
global dict
dict = [1,2,3,4,5,6,7,8,9,10,10,10,10];

global randomPI
randomPI=zeros(10,2,10,2)+0.5;
 
epsilon=0.1;
alpha=0.1;

% choose initial PI from random PI or any generated PI
initialPI=generate_initialPI();
PI=initialPI;
 
MC_control(PI,epsilon,20000000);
% Sarsa(PI,epsilon,alpha,20000000);
% QLearning(PI,epsilon,alpha,20000000);

 
function MC_control(PI,epsilon,iter)
global randomPI
win_count=0;
draw_count=0;
lose_count=0;
 
Nsa=zeros(10,2,10,2);
Qsa=zeros(10,2,10,2);
 
for i=1:1:iter
    [reward,salist]=onegame(PI);
    
    if reward==1
        win_count = win_count+1;
    elseif reward==-1
        lose_count = lose_count+1;
    else
        draw_count = draw_count+1;
    end
 
    if ~isempty(salist)
        [Nsa,Qsa]=updateQsa_MC(reward,salist,Nsa,Qsa);
    end
    % epsilon=1/i;
    greedyPI=greedy(Qsa);
    PI=epsilon_greedy(epsilon,greedyPI,randomPI);
    if mod(i,20000)==0
        plotPI(greedyPI);
        drawnow;
        
        win_rate=win_count/200;
        draw_rate=draw_count/200;
        lose_rate=lose_count/200;
        
        fprintf("simulated %d times\n", i)
        fprintf("win:  %0.2f%%\n", win_rate);
        fprintf("draw: %0.2f%%\n", draw_rate);
        fprintf("lose: %0.2f%%\n", lose_rate);
        
        win_count=0;
        draw_count=0;
        lose_count=0;
    end
end
printPI(greedyPI);
end

function Sarsa(PI,epsilon,alpha,iter)
global randomPI
win_count=0;
draw_count=0;
lose_count=0;
 
Qsa=zeros(10,2,10,2);

for i=1:1:iter
    [reward,salist]=onegame(PI);
    
    if reward==1
        win_count = win_count+1;
    elseif reward==-1
        lose_count = lose_count+1;
    else
        draw_count = draw_count+1;
    end
    
    alpha=0.4*20000/(i+20000);
    epsilon=1000000/(i+1000000);
 
    if ~isempty(salist)
        Qsa=updateQsa_Sarsa(reward,salist,Qsa,alpha);
    end
    
    greedyPI=greedy(Qsa);
    PI=epsilon_greedy(epsilon,greedyPI,randomPI);
    if mod(i,20000)==0
        plotPI(greedyPI);
        drawnow;
        
        win_rate=win_count/200;
        draw_rate=draw_count/200;
        lose_rate=lose_count/200;
        
        fprintf("simulated %d times\n", i)
        fprintf("win:  %0.2f%%\n", win_rate);
        fprintf("draw: %0.2f%%\n", draw_rate);
        fprintf("lose: %0.2f%%\n", lose_rate);
        
        win_count=0;
        draw_count=0;
        lose_count=0;
    end
end
printPI(greedyPI);
end

function QLearning(PI,epsilon,alpha,iter)
global randomPI
win_count=0;
draw_count=0;
lose_count=0;
 
Qsa=zeros(10,2,10,2);

for i=1:1:iter
    [reward,salist]=onegame(PI);
    
    if reward==1
        win_count = win_count+1;
    elseif reward==-1
        lose_count = lose_count+1;
    else
        draw_count = draw_count+1;
    end
    
    alpha=0.4*2000/(i+2000);
    epsilon=800000/(i+800000);
 
    if ~isempty(salist)
        Qsa=updateQsa_QLearning(reward,salist,Qsa,alpha);
    end
    
    greedyPI=greedy(Qsa);
    PI=epsilon_greedy(epsilon,greedyPI,randomPI);
    if mod(i,20000)==0
        plotPI(greedyPI);
        drawnow;
        
        win_rate=win_count/200;
        draw_rate=draw_count/200;
        lose_rate=lose_count/200;
        
        fprintf("simulated %d times\n", i)
        fprintf("win:  %0.2f%%\n", win_rate);
        fprintf("draw: %0.2f%%\n", draw_rate);
        fprintf("lose: %0.2f%%\n", lose_rate);
        
        win_count=0;
        draw_count=0;
        lose_count=0;
    end
end
printPI(greedyPI);
end
 
function PI=epsilon_greedy(epsilon,greedyPI,randomPI)
if epsilon>1
    epsilon=1;
end
PI=epsilon*randomPI+(1-epsilon)*greedyPI;
end
 
function PI=greedy(Qsa)
PI=zeros(10,2,10,2);
for i=1:1:10
    for j=1:1:2
        for k=1:1:10
            if Qsa(i,j,k,1)>Qsa(i,j,k,2)
                PI(i,j,k,1)=1;
            else
                PI(i,j,k,2)=1;
            end
        end
    end
end
end
 
function [Nsa,Qsa]=updateQsa_MC(reward,salist,Nsa,Qsa)
s=size(salist);
for i=1:1:s(2)
    sa=salist{i};
    k=sa+[-11 1 0 1];
    Nsa(k(1),k(2),k(3),k(4))=Nsa(k(1),k(2),k(3),k(4))+1;
%     N=Nsa(k(1),k(2),k(3),k(4));
%     Q=Qsa(k(1),k(2),k(3),k(4));
%     Qsa(k(1),k(2),k(3),k(4))=Q+(reward-Q)/N;
    Qsa(k(1),k(2),k(3),k(4))=Qsa(k(1),k(2),k(3),k(4))+(reward-Qsa(k(1),k(2),k(3),k(4)))/Nsa(k(1),k(2),k(3),k(4));
end
end

function Qsa=updateQsa_Sarsa(reward,salist,Qsa,alpha)
s=size(salist);
for i=1:1:s(2)
    k=salist{i}+[-11 1 0 1];
    Q_t0=Qsa(k(1),k(2),k(3),k(4));
    if i<s(2)
        p=salist{i+1}+[-11 1 0 1];
        Q_t1=Qsa(p(1),p(2),p(3),p(4));
        Qsa(k(1),k(2),k(3),k(4))=Q_t0+alpha*(Q_t1-Q_t0);
    else
        Qsa(k(1),k(2),k(3),k(4))=Q_t0+alpha*(reward-Q_t0);
    end
end
end

function Qsa=updateQsa_QLearning(reward,salist,Qsa,alpha)
s=size(salist);
for i=1:1:s(2)
    k=salist{i}+[-11 1 0 1];
    Q_t0=Qsa(k(1),k(2),k(3),k(4));
    if i<s(2)
        p=salist{i+1}+[-11 1 0 1];
        Q_t1=max(Qsa(p(1),p(2),p(3),:));
        Qsa(k(1),k(2),k(3),k(4))=Q_t0+alpha*(Q_t1-Q_t0);
    else
        Qsa(k(1),k(2),k(3),k(4))=Q_t0+alpha*(reward-Q_t0);
    end
end
end

function [reward,salist]=onegame(PI)
global dict;
 
salist={};
 
player_cards = dict(randi(13,1,2));
dealer_cards = dict(randi(13,1,2));
 
player_hand=initial_cards(player_cards);
dealer_hand=initial_cards(dealer_cards);
shown_card=dealer_cards(1);
 
if natural(player_hand)
    %fprintf("player natural\n");
    if natural(dealer_hand)
        reward=0;
        %fprintf("dealer natural\n");
    else
        reward=1;
    end
    return;
end
 
[player_bust,salist]=players_turn(player_hand,shown_card,PI);
if player_bust
    reward=-1;
    %fprintf("player bust\n");
    return;
end
 
[dealer_bust,dealer_finalsum]=dealers_turn(dealer_hand);
if dealer_bust
    reward=1;
    %fprintf("dealer bust\n");
    return;
end
 
player_finalsum=salist{end}(1);
 
if player_finalsum>dealer_finalsum
    reward=1;
elseif player_finalsum<dealer_finalsum
    reward=-1;
else
    reward=0;
end
%fprintf("player finalsum: %d\n",player_finalsum);
%fprintf("dealer finalsum: %d\n",dealer_finalsum);
end
 
function [bust,salist]=players_turn(hand,shown_card,PI)
bust=0;
salist={};
while hand(1)<12
    hand=hit(hand);
end
action=1;
while action==1
    state=[hand shown_card];
    action=decision(state,PI);
    sa=[state action];
    salist{end+1}=sa;
    if action == 1
        hand=hit(hand);
    end
    if ifbust(hand)
        bust=1;
        action=0;
    end
end
% disp(salist);
end
 
function [bust,finalsum]=dealers_turn(hand)
bust=0;
while hand(1)<17
    hand=hit(hand);
end
if ifbust(hand)
    bust=1;
end
finalsum=hand(1);
end
 
function b=decision(state,PI)
index=state+[-11 1 0];
%pmf=squeeze(PI(index(1),index(2),index(3),:));
p1=PI(index(1),index(2),index(3),1);
b=rand(1)>p1;
end
 
function new_hand=hit(hand)
global dict;
card=dict(randi(13,1));
new_hand=addcard(hand, card);
%fprintf("hit\n");
end
 
function b=validstate(hand)
b=0;
sum=hand(1);
if sum>=12 && sum <=21
    b=1;
end
end
 
function hand=initial_cards(cards)
hand=[0, 0];
hand=addcard(hand,cards(1));
hand=addcard(hand,cards(2));
end
 
function b=ifbust(hand)
b = hand(1) > 21 && hand(2) == 0;
end
 
function b=natural(hand)
b = hand(1) == 21;
end
 
function new_hand=addcard(hand, card)
if ifbust(hand)
    new_hand=hand;
    return;
end
if card~=1
    hand(1)=hand(1)+card;
    if hand(1)>21 && hand(2)==1
        hand(2)=0;
        hand(1)=hand(1)-10;
    end
else
    if hand(2)==1              % if there is one usable ace, new ace should be counted as 1
        hand(1)=hand(1)+1;
    else
        hand(1)=hand(1)+11;
        hand(2)=hand(2)+1;
    end
    if hand(1)>21 && hand(2)==1
        hand(1)=hand(1)-10;
        hand(2)=hand(2)-1;
    end
end
new_hand=hand;
% disp(hand);
end


function initialPI=generate_initialPI()
initialPI=zeros(10,2,10,2);
initialPI(9,:,:,1)=1;
initialPI(10,:,:,1)=1;
for i=1:1:8
    initialPI(i,:,:,2)=1;
end
end


function printPI(PI)
PI1=squeeze(PI(:,1,:,2));
PI2=squeeze(PI(:,2,:,2));
disp(PI1);
disp(PI2);
end
 

function plotPI(PI)
PI1=squeeze(PI(:,1,:,2));
PI2=squeeze(PI(:,2,:,2));
x=[1 10];
y=[12 21];
clims=[0 1];
 
subplot(1,2,1);
imagesc(x,y,PI2,clims);
set(gca,'YDir','normal');
title('Usable ace');
xlabel('Dealer showing');
ylabel('Player sum');
xticks(1:10);
yticks(12:21);
axis square;
 
subplot(1,2,2);
imagesc(x,y,PI1,clims);
set(gca,'YDir','normal');
title('No usable ace');
xlabel('Dealer showing');
ylabel('Player sum');
xticks(1:10);
yticks(12:21);
axis square;
end