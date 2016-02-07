#include "BehaveRT.h"


#define UTILITYLOGGER_MSGMAXSIZE 100

UtilityLogger::UtilityLogger( unsigned int maxSize )
{
	mMaxSize = maxSize;
	mModified = false;
	mLevel = LOG_INFO;
}

UtilityLogger::~UtilityLogger(void)
{
	mMsgList.clear();
}

void UtilityLogger::log( msgSenderType msgSender, msgTextType msgText, mLogLevels level )
{
	// Se viene raggiunta la dimensione massima
	// elimina il primo elemento inserito  
	if (mMsgList.size() > mMaxSize)
		mMsgList.pop_front();
	
	logItem item = make_pair( 
			make_pair ( 
				msgSender, 
				msgText ), level);

	// Write on standard output
	cout << msgSender << ">> " << getMsgString(item) << std::endl;

	// Creazione del messaggio strutturato
	mMsgList.push_back( item );

	mModified = true;
}


void UtilityLogger::quickLog(msgSenderType msgSender, const char *fmt, ...)
{
	char msg[UTILITYLOGGER_MSGMAXSIZE];

	va_list ap;
	va_start(ap, fmt);
	//ap += 4;
		
	sprintf(msg, fmt, ap);
	
	va_end(ap);

	log(msgSender, msg);
}

bool UtilityLogger::checkLevel(logItem item)
{
	return (item.second == mLevel || item.second == LOG_INFO);
}

bool UtilityLogger::checkSender(logItem item)
{
	return (item.first.first == mSenderFilter);
}


string UtilityLogger::getMsgString(logItem item)
{
	// first contiene il messaggio
	// first.second contiene il corpo del messaggio
	return item.first.second;
}


void UtilityLogger::setLevelString( string level )
{
	if (level == "debug")
		mLevel = LOG_DEBUG;
	else
		mLevel = LOG_INFO;
}


string UtilityLogger::getMsgListString( string separator )
{
	string logStr;

	for (mMsgList_it = mMsgList.begin(); 
		mMsgList_it != mMsgList.end(); ++mMsgList_it)
	{
		logItem item = *mMsgList_it;

		// Il filtro in base al sender consente di visualizzare solo i messaggi
		// il cui sender è uguale a mSenderFilter
		if ( mSenderFilter != "" )
			if ( !checkSender( item ) )
				continue;
			
		logStr += item.first.first + ": " + item.first.second + separator;
	}

	return logStr;
}